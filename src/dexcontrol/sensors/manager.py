# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Sensor manager for managing robot sensors.

This module provides the Sensors class that manages multiple sensor instances
based on configuration and provides unified access to them.
"""

import time
from typing import TYPE_CHECKING, Any

from dexbot_utils.configs import BaseComponentConfig
from loguru import logger

if TYPE_CHECKING:
    from dexcontrol.sensors.camera.usb_camera import USBCameraSensor
    from dexcontrol.sensors.camera.zed_camera import (
        ZedCameraSensor,
        ZedXOneCameraSensor,
    )
    from dexcontrol.sensors.imu.chassis_imu import ChassisIMUSensor
    from dexcontrol.sensors.imu.zed_imu import ZedIMUSensor
    from dexcontrol.sensors.lidar.rplidar import RPLidarSensor
    from dexcontrol.sensors.ultrasonic import UltrasonicSensor


class Sensors:
    """Manages robot sensors including cameras, IMUs, LiDAR, and ultrasonic sensors.

    Provides a unified interface for creating, managing, and shutting down all types of
    sensors used by the robot. Instantiates sensors based on configuration and provides
    access to them as attributes.

    Attributes:
        _sensors: List of instantiated sensor objects.
    """

    if TYPE_CHECKING:
        # Type annotations for dynamically created sensor attributes
        head_camera: ZedCameraSensor
        left_wrist_camera: ZedXOneCameraSensor
        right_wrist_camera: ZedXOneCameraSensor
        base_left_camera: USBCameraSensor
        base_right_camera: USBCameraSensor
        base_front_camera: USBCameraSensor
        base_back_camera: USBCameraSensor
        base_imu: ChassisIMUSensor
        head_imu: ZedIMUSensor
        lidar: RPLidarSensor
        ultrasonic: UltrasonicSensor

    def __init__(self, configs: dict[str, BaseComponentConfig]) -> None:
        """Initialize sensors from configuration.

        Args:
            configs: Configuration for all sensors (VegaSensorsConfig or DictConfig).
        """
        self._sensors: list[Any] = []
        self._shutdown_sensor_ids: set[int] = set()
        self._config_sensor_names: set[str] = set(configs.keys())
        try:
            for sensor_name, sensor_config in configs.items():
                sensor = self._create_sensor(sensor_config, str(sensor_name))
                if sensor is not None:
                    setattr(self, str(sensor_name), sensor)
                    if not any(existing is sensor for existing in self._sensors):
                        self._sensors.append(sensor)
        except BaseException:
            while True:
                try:
                    self.shutdown()
                    break
                except BaseException as exc:  # pylint: disable=broad-except
                    logger.error(
                        "Sensor constructor cleanup is not yet safe; retrying: {}",
                        exc,
                    )
                    time.sleep(0.1)
            raise

    def __getattr__(self, name: str) -> Any:
        """Provide clear error messages when accessing unavailable sensors.

        This method is only called when normal attribute lookup fails, meaning
        the sensor was not initialized (disabled or not present on this robot).

        Args:
            name: The attribute name being accessed.

        Raises:
            SensorNotAvailableError: If the name is a known sensor that
                is not available on this robot.
            AttributeError: If the name is not a known sensor.
        """
        from dexcontrol.exceptions import SensorNotAvailableError

        try:
            known = object.__getattribute__(self, "_config_sensor_names")
        except AttributeError:
            known = set()
        if name in known:
            raise SensorNotAvailableError(name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def has_sensor(self, name: str) -> bool:
        """Check if a sensor is initialized and available.

        Args:
            name: The sensor name (e.g., "head_imu", "ultrasonic").

        Returns:
            True if the sensor exists and was successfully initialized.
        """
        try:
            sensor = object.__getattribute__(self, name)
            return sensor is not None
        except AttributeError:
            return False

    def _create_sensor(self, config: BaseComponentConfig, name: str) -> Any | None:
        """Creates and initializes a sensor from config.

        Args:
            config: Sensor configuration.
            name: Name of the sensor for logging.

        Returns:
            Initialized sensor object or None if creation fails.
        """
        from dexcontrol.core.config import get_sensor_mapping
        if not config.enabled:
            logger.debug(f"Sensor {name} is disabled")
            return None

        sensor_mapping = get_sensor_mapping()
        if type(config) not in sensor_mapping:
            return None

        sensor_class = sensor_mapping[type(config)]
        # Retain ownership before __init__. Multi-stream sensors can acquire an
        # early subscriber and then fail while creating a later one.
        sensor = sensor_class.__new__(sensor_class)
        self._sensors.append(sensor)
        try:
            sensor_class.__init__(sensor, name=name, configs=config)
        except BaseException:
            self._shutdown_sensor_until_safe(sensor)
            self._sensors.remove(sensor)
            raise

        try:
            if hasattr(sensor, "start"):
                sensor.start()
        except BaseException:
            self._shutdown_sensor_until_safe(sensor)
            self._sensors.remove(sensor)
            raise

        return sensor

    @staticmethod
    def _shutdown_sensor_until_safe(sensor: Any) -> None:
        shutdown = getattr(sensor, "shutdown", None)
        if not callable(shutdown):
            return
        while True:
            try:
                shutdown()
                return
            except BaseException as cleanup_exc:
                logger.error(
                    "Sensor rollback is not yet safe; retrying: {}", cleanup_exc
                )
                time.sleep(0.1)

    def shutdown(self) -> None:
        """Shuts down all active sensors."""
        logger.info("Shutting down all sensors...")
        errors = []
        for sensor in self._sensors:
            sensor_id = id(sensor)
            if sensor_id in self._shutdown_sensor_ids:
                continue
            try:
                if hasattr(sensor, "shutdown"):
                    sensor.shutdown()
                    self._shutdown_sensor_ids.add(sensor_id)
                    # Small delay between each sensor shutdown to prevent race conditions
                    time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error shutting down sensor: {e}")
                errors.append(str(e))

        if errors:
            raise RuntimeError("; ".join(errors))

        # Additional delay to ensure all sensor subscribers have been undeclared
        time.sleep(0.1)
        logger.info("All sensors shut down")

    def get_active_sensors(self) -> list[str]:
        """Gets list of active sensor names.

        Returns:
            List of sensor names that are currently active.
        """
        excluded_attrs = {'shutdown', 'get_active_sensors', 'wait_for_sensors',
                          'wait_for_all_active', 'get_sensor_count', 'has_sensor'}
        active_sensors = []

        for attr_name in dir(self):
            if (not attr_name.startswith('_') and
                attr_name not in excluded_attrs):
                sensor = getattr(self, attr_name)
                if (sensor is not None and
                    hasattr(sensor, 'is_active') and
                    sensor.is_active()):
                    active_sensors.append(attr_name)
        return active_sensors

    def wait_for_sensors(self, timeout: float = 5.0) -> bool:
        """Waits for all enabled sensors to become active.

        Args:
            timeout: Maximum time to wait for sensors to become active.

        Returns:
            True if all sensors became active, False otherwise.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if all(sensor.is_active() for sensor in self._sensors
                  if hasattr(sensor, 'is_active')):
                logger.info("All sensors are active")
                return True
            time.sleep(0.1)

        logger.warning(f"Not all sensors became active within {timeout}s")
        return False

    def wait_for_all_active(self, timeout: float = 5.0) -> bool:
        """Waits for all enabled sensors to become active.

        Alias for wait_for_sensors() for backward compatibility.

        Args:
            timeout: Maximum time to wait for sensors to become active.

        Returns:
            True if all sensors became active, False otherwise.
        """
        return self.wait_for_sensors(timeout)

    def get_sensor_count(self) -> int:
        """Gets the total number of instantiated sensors.

        Returns:
            Number of sensors that were successfully created.
        """
        return len(self._sensors)
