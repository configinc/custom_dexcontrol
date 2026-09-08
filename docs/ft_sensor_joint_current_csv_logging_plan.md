# 계획: F/T 센서 + Joint Current CSV 로깅

## 목표

`robot-control-interface`에서 **Start Inference**를 눌러 recording이 시작된 시점부터
**Stop Inference**(또는 에러로 인한 recording 중단)로 recording이 끝나는 시점까지,
매 control step마다 아래 값을 하나의 CSV 파일에 기록한다.

1. F/T 센서: `Fx, Fy, Fz, Tx, Ty, Tz`
2. 각 joint의 current (전류, A)

## 현재 상태 조사 결과

### 1) F/T 센서 — 이미 구현되어 있음

- `Arm.wrench_sensor` (`src/dexcontrol/core/arm.py:99-109`, `ArmWrenchSensor` 클래스는 같은 파일
  779-868줄)가 `get_wrench_state() -> np.ndarray` (shape `(6,)`, `[fx, fy, fz, tx, ty, tz]`)를
  제공한다.
- `VegaRobot.get_robot_state()` (`src/dexcontrol/core/vega/robot.py:932-965`)가 이미
  `state_dict["wrench_state"]`로 이 값을 채워 넣고 있고, gRPC observation
  (`src/dexcontrol/core/robotenv_vega/server.py:368-373`, `:730-731`)까지 전달된다.
- `robot-control-interface`의 `RobotEnvClient._parse_observation()`은 이 `wrench_state`를
  proto→dict로 그대로 통과시켜 `obs["wrench_state"]`에 넣지만, **`RobotEnvClient._build_robot_state()`
  (`robot-control-interface/core/robot_env_client.py:412-460`)가 이를 다시 걸러내고 있어
  최종 `robot_state`에는 포함되지 않는다.** → 여기 한 곳만 고치면 F/T 값은 바로 사용 가능.

### 2) Joint current — API는 있지만 firmware 값이 채워지는지 미확인 (블로커)

- `RobotComponent.get_joint_current()` / `get_joint_current_dict()`
  (`src/dexcontrol/core/component.py:414-458`)는 `state["cur"]`를 읽는다.
- 반면 실제로 파이프라인에 연결되어 있는 것은 `get_joint_torque()`
  (`state["torque"]`, Nm)이며, `VegaRobot.get_robot_state()`가 로깅하는 것도 torque다.
  즉 **현재는 "current"가 아니라 "torque"가 로깅되고 있다.**
- 사용자가 작성 중인 `examples/troubleshooting/check_current_torque_fields.py`가
  정확히 이 문제(“cur”, “torque” 필드가 firmware에서 실제로 채워지는지)를 확인하는
  스크립트다. **이 스크립트를 실제 로봇에서 먼저 실행해서 `"cur"` 필드가 채워지는지
  확인하는 것이 이 계획의 선행 조건이다.**
  - `"cur"`가 채워진다면: `get_joint_current()`를 그대로 사용.
  - `"cur"`가 비어있거나 없다면: 두 가지 선택지
    1. torque(Nm)를 대신 CSV에 기록하고 컬럼명을 명확히 한다(`torque_j1`처럼), 또는
    2. firmware/저수준 드라이버에 current 필드를 채우는 작업이 별도로 필요함을
       문서화하고, 이번 스코프에서는 torque로 대체한다.
  - 이 문서의 나머지 부분은 `"cur"`가 채워진다는 가정 하에 작성하되, 안 채워질 경우의
    fallback을 명시한다.

### 3) 기존 CSV/로깅 인프라 — 없음, 참고할 만한 선례만 존재

- `csv` 모듈이나 pandas를 쓰는 기존 로거는 없다.
- 가장 가까운 선례는 `VegaRobot`의 `VEL_LOG_PATH` 환경변수 기반 수동 파일 로거
  (`src/dexcontrol/core/vega/robot.py:225-235`, `:899-902`, `:1330-1332`): env var로
  경로를 받아 append 모드로 열고, 헤더 한 줄 쓴 뒤 매 control step마다 콤마로 join한
  한 줄을 쓰고, shutdown 시 닫는다. 이번 기능도 같은 스타일(표준 `csv` 모듈 정도만 추가)로
  구현하는 것을 권장.

### 4) Recording start/stop 트리거 — `robot-control-interface`, REST 기반

- `POST /start-inference` (`policy_inference_interface/main.py:5921`)가
  외부 recorder 서비스에 `POST {RECORDER_URL}/start-recording`을 호출한 뒤
  `PolicyManager.start_inference`를 실행한다.
- `POST /stop-inference` (`main.py:6424`)가 `PolicyManager.stop_inference` 후
  `_stop_recorder_recording()` (`main.py:2310-2345`)을 호출한다. 이 함수는 여러 에러/조기
  중단 경로(`main.py:6330,6364,6379,6386,6392,6404,6417`)에서도 방어적으로 호출된다.
- **pub/sub이나 콜백/훅 시스템은 없다.** 순수 REST 요청/응답이므로, 새 CSV 로거의
  시작/종료 호출은 이 지점들에 직접 추가해야 한다.
- 실제 recorder(영상 등)는 이 두 레포에 소스가 없는 완전히 별도 서비스
  (`localhost:8000`)다. 우리가 만들 F/T·current CSV 로거는 이 외부 recorder와는
  독립적으로, `robot-control-interface` 프로세스 안에서 직접 관리한다.
- Per-step 데이터가 만들어지는 위치는 `PolicyRunner`의 control loop
  (`policy_runner.py:3148-3210`)이며, `action_info1/2["robot_state"]`가
  `RobotEnvClient._build_robot_state()`의 결과물이다. 현재 `_log_inference_step()`
  (`policy_runner.py:1631-1634`)은 no-op으로 비활성화되어 있음 — 이 자리가
  per-step 파일 로깅을 넣기에 가장 자연스러운 위치.

## 설계

### 데이터 흐름

```
[Vega firmware]
   → dexcontrol: ArmWrenchSensor.get_wrench_state() / get_joint_current()
   → VegaRobot.get_robot_state()  (custom_dexcontrol, 이미 존재/일부 확장 필요)
   → gRPC RobotEnv observation    (robotenv_vega/server.py, wrench는 존재, current는 추가 필요할 수 있음)
   → RobotEnvClient._parse_observation()   (robot-control-interface, 이미 통과시킴)
   → RobotEnvClient._build_robot_state()   (robot-control-interface, wrench/current 통과하도록 수정 필요)
   → PolicyRunner 제어 루프의 action_info["robot_state"]
   → [신규] FTCurrentCsvLogger.log_step(...)   (robot-control-interface, 신규)
```

Recording 시작/종료는 `main.py`의 `/start-inference`, `/stop-inference`(및 조기 중단 경로)에서
`FTCurrentCsvLogger`의 `start()` / `stop()`을 직접 호출해 제어한다.

### CSV 스키마

한 에피소드(recording 1회)당 CSV 파일 하나. 파일명 예:
`ft_current_{episode_id 또는 timestamp}.csv`

| 컬럼 | 설명 |
| --- | --- |
| `timestamp` | 각 step의 wall-clock 또는 monotonic time (초, float) |
| `step_index` | 0부터 증가하는 정수 |
| `fx, fy, fz, tx, ty, tz` | 좌/우 팔에 F/T 센서가 각각 있으므로 `left_fx...left_tz`, `right_fx...right_tz`로 분리 (센서가 한쪽만 있으면 없는 쪽은 빈 값 또는 컬럼 생략) |
| `cur_{joint_name}` | 각 joint 전류(A). Joint 이름은 `RobotComponent.get_joint_name()`을 그대로 사용 (예: `L_arm_j1`..`L_arm_j7`, `R_arm_j1`..`R_arm_j7`, 그리고 필요 시 다리/허리/헤드 등 다른 컴포넌트도 포함할지 결정 필요 — 기본 스코프는 양팔로 한정할지 사용자 확인 필요) |

> 미확인 항목(아래 "확인 필요" 참고): current 대신 torque만 쓸 경우 컬럼명을
> `torque_{joint_name}`으로 바꾸고 단위를 Nm로 문서화한다.

### 구현 단계

**Phase 0 — 선행 확인 (코드 변경 없음)**
1. 실제 로봇에서 `examples/troubleshooting/check_current_torque_fields.py` 실행,
   `"cur"` 필드 채워지는지 확인. 결과에 따라 아래 Phase 1의 세부 사항 확정.

**Phase 1 — `custom_dexcontrol` (필요한 경우에만)**
1. (`"cur"`가 이미 채워진다면 스킵) `VegaRobot.get_robot_state()`
   (`src/dexcontrol/core/vega/robot.py:932-965`)에 `joint_currents` 필드를 추가해
   `wrench_state`와 같은 방식으로 채운다.
2. gRPC observation 스펙(`src/dexcontrol/core/robotenv_vega/server.py:368-373`,
   `:730-731` 부근)에 `joint_currents`용 `float_array` 필드를 wrench_state와 동일한
   패턴으로 추가한다.
3. 관련 examples/tests 업데이트, `check_current_torque_fields.py`는 정식
   troubleshooting 예제로 커밋.

**Phase 2 — `robot-control-interface`**
1. `RobotEnvClient._build_robot_state()` (`core/robot_env_client.py:412-460`)를 수정해
   `wrench_state`(및 Phase 1에서 추가했다면 `joint_currents`)를 최종 `robot_state`
   dict에 포함시킨다.
2. 신규 모듈(예: `core/ft_current_csv_logger.py`)에 `FTCurrentCsvLogger` 클래스 구현:
   - `start(output_path)`: CSV 파일 open, 표준 `csv.writer`로 헤더 작성.
   - `log_step(robot_state, step_index, timestamp)`: 한 행 작성.
   - `stop()`: flush & close.
   - 에러 시에도 안전하게 close 되도록 처리 (`VEL_LOG_PATH` 로거의 shutdown 패턴 참고).
3. `PolicyRunner`의 control loop(`policy_runner.py:3148-3210`, 특히 현재 no-op인
   `_log_inference_step()` 자리, `policy_runner.py:1631-1634`)에서 매 step마다
   `FTCurrentCsvLogger.log_step(...)` 호출.
4. `main.py`의 `/start-inference` 핸들러(`:5921` 부근, recorder
   start 호출 직후)에서 `FTCurrentCsvLogger.start(...)` 호출.
5. `main.py`의 `/stop-inference` 핸들러(`:6424`) 및 `_stop_recorder_recording()`이
   호출되는 모든 조기 중단 경로(`:6330,6364,6379,6386,6392,6404,6417`)에서
   `FTCurrentCsvLogger.stop()`을 함께 호출해, recording이 어떤 경로로 끝나든 CSV가
   확실히 닫히도록 한다.
6. CSV 파일 저장 경로: recorder가 쓰는 episode 디렉토리 규칙을 따를지, 별도 경로로
   둘지 확인 필요(아래 "확인 필요" 참고).

**Phase 3 — 검증**
1. 실제(또는 시뮬레이션) 로봇에서 Start → 몇 초간 팔 움직임/외력 인가 → Stop 흐름을
   실행해 CSV가 생성되고, 값이 그럴듯한지(F/T가 외력에 반응하는지, current/torque가
   0이 아닌지) 확인.
2. 에러로 inference가 조기 중단되는 경로(예: 안전 정지)에서도 CSV가 정상적으로
   닫히는지 확인.
3. 좌/우 팔 중 하나에만 F/T 센서가 없는 로봇 구성에서도 죽지 않고 빈 값으로
   처리되는지 확인.

## 구현 노트 (실제 반영된 내용)

계획 대비 실제 구현에서 확정/변경된 부분:

- **Trigger 위치**: `main.py`의 `/start-inference`/`/stop-inference`를 직접 건드리지 않고,
  이미 존재하던 `PolicyRunner._setup_inference_logging()` / `_finalize_inference_logging()`
  훅(원래 비활성화된 legacy per-step 로거를 위해 있던 지점)에 CSV logger의 `start()`/`close()`를
  붙였다. 이 두 함수는 정상 종료·에러·조기 취소 등 `_run_gr00t_inference`의 모든 종료 경로에서
  이미 호출되고 있어, `main.py` 쪽 8곳의 `_stop_recorder_recording()` 호출부를 일일이 건드리는
  것보다 diff가 훨씬 작고 안전하다.
- **Current vs Torque**: `VegaRobot.get_robot_state()`에 `get_joint_current()`를 호출해
  `"cur"`가 없으면(`ValueError`) 0으로 채우는 방식으로 구현(torque와 동일한 패턴). 즉 하드웨어
  검증 없이도 안전하게 동작하며, `check_current_torque_fields.py`로 `"cur"`가 실제 채워지는 게
  확인되면 코드 변경 없이 바로 실제 값이 CSV에 찍힌다. 계속 0이면 firmware 쪽 확인이 필요하다는 신호.
- **Joint 범위**: 양팔(`L_arm`, `R_arm` 각 7개)로 한정 — F/T 센서와 동일한 스코프.
- **CSV 경로/파일명**: `{output_dir}/ft_current_{run_id}.csv` (`output_dir`는 기존
  `save_results`가 쓰는 것과 동일하게 `config.get("output_dir", "./output")`, `run_id`는
  세션마다 이미 발급되는 `self.run_id` 재사용).
- **샘플링 레이트**: control loop 매 step (`_log_inference_step` 호출 시점, 기존에는 no-op이었음).
- **편측 F/T 센서 없음**: 커스텀 처리 없이 기존 관행대로 0.0으로 채워짐 (해당 팔에
  `wrench_sensor`가 없으면 `get_robot_state()`가 이미 zeros(6)를 반환).

변경 파일:
- `custom_dexcontrol/src/dexcontrol/core/vega/robot.py` — `get_robot_state()`에 `joint_currents` 추가.
- `custom_dexcontrol/src/dexcontrol/core/robotenv_vega/server.py` — gRPC observation에 `joint_currents` 필드 추가.
- `robot-control-interface/core/robot_env_client.py` — `_build_robot_state()`에서 `wrench_state`/`joint_currents` pass-through.
- `robot-control-interface/core/ft_current_csv_logger.py` (신규) — CSV writer.
- `robot-control-interface/policy_inference_interface/policy_runner.py` — logger 생성/시작/종료/per-step 기록 연결.

## 확인 필요 (사용자 결정 필요)

1. **Current vs Torque**: `"cur"` 필드가 firmware에서 실제로 채워지는지. 안 채워지면
   torque로 대체할지, firmware 쪽 작업을 별도로 진행할지.
2. **로깅 대상 joint 범위**: 양팔(`L_arm_j1-7`, `R_arm_j1-7`)만 포함할지, 그리퍼/헤드/허리/다리
   등 다른 컴포넌트의 current도 포함할지.
3. **CSV 저장 위치/파일명 규칙**: 외부 recorder가 만드는 episode 디렉토리와 같은 곳에
   둘지, `robot-control-interface`가 별도로 관리하는 경로를 새로 둘지.
4. **샘플링 레이트**: control loop의 매 step(현재 policy control_hz)마다 기록할지,
   별도 다운샘플링이 필요한지.
5. **양팔 중 한쪽만 F/T 센서가 있는 경우**의 컬럼 처리 방식(컬럼 생략 vs 빈 값).
