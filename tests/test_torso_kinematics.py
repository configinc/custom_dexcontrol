from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "dexcontrol"))
from core.robotenv_vega.torso_kinematics import TorsoKinematics


@pytest.fixture
def model():
    return TorsoKinematics([[-.235,.248],[.396,.082],[-.40718,.09764],[-.05908,.44528]],
                           [[0,1.57],[0,3.141],[-1.57,1.57]],[.2]*3)


def test_fk_ik_roundtrip_with_joint_limits(model):
    for q in ([.8,1.6,.8],[.3,.9,-.2],[1.4,2.5,.3]):
        result=model.inverse(model.forward(q), q)
        np.testing.assert_allclose(result,q,atol=1e-10)


@pytest.mark.parametrize("axis", range(3))
def test_each_shoulder_axis_preserves_the_other_two(model,axis):
    q=np.array([.8,1.6,.8]);start=model.forward(q)
    goal=start.copy();goal[axis]+=.03 if axis<2 else .05
    path=model.plan(q,goal)
    for elapsed in np.arange(.02,path.duration+.02,.02):
        next_q, velocity=path.sample(elapsed)
        assert np.all(np.abs(next_q-q)<=model.velocity_limits*.02+1e-10)
        assert np.all(np.abs(velocity)<=model.velocity_limits+1e-10)
        assert np.all(next_q>=model.limits[:,0]) and np.all(next_q<=model.limits[:,1])
        np.testing.assert_allclose(np.delete(model.forward(next_q),axis),np.delete(start,axis),atol=1e-5)
        q=next_q
    np.testing.assert_allclose(model.forward(q),goal,atol=1e-9)


def test_unreachable_or_out_of_limit_goal_is_rejected(model):
    with pytest.raises(ValueError):model.inverse([10,10,0],[.8,1.6,.8])
    with pytest.raises(ValueError):model.inverse(model.forward([-1,1,0]),[.8,1.6,.8])


def test_urdf_limits_intersect_component_limits(tmp_path):
    xml='<robot name="test">'
    parents=['base','torso_l1','torso_l2','torso_l3'];children=['torso_l1','torso_l2','torso_l3','arm_center']
    for i,name in enumerate(['torso_j1','torso_j2','torso_j3','arm_center_j0']):
        kind='revolute' if i<3 else 'fixed'
        xml+=f'<joint name="{name}" type="{kind}"><parent link="{parents[i]}"/><child link="{children[i]}"/><origin xyz="0.1 0 0.2" rpy="0 0 0"/>'
        if i<3:xml+=f'<axis xyz="0 {[-1,1,-1][i]} 0"/><limit lower="0" upper="1" velocity="0.03"/>'
        xml+='</joint>'
    p=tmp_path/'model.urdf';p.write_text(xml+'</robot>')
    model=TorsoKinematics.from_urdf(p,[[.1,.9]]*3,[.02]*3)
    np.testing.assert_allclose(model.limits,[[.1,.9]]*3)
    np.testing.assert_allclose(model.velocity_limits,[.02]*3)
