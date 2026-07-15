from abcdl.mcap import schemas


def test_robotstate_fields():
    m = schemas.RobotState()
    m.position.extend([0.0] * 6)
    m.velocity.extend([0.0] * 7)
    m.torque.extend([0.0] * 7)
    m.pose.extend([0.0] * 16)
    assert len(m.position) == 6 and len(m.pose) == 16


def test_gripperstate_and_instructions():
    g = schemas.GripperState()
    g.position.extend([0.5])
    assert g.position[0] == 0.5
    ins = schemas.Instructions(data="open the umbrella")
    assert ins.data == "open the umbrella"


def test_annotation():
    a = schemas.Annotation(data="subtask: grab")
    assert a.data == "subtask: grab"
