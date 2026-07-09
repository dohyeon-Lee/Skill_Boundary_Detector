import abcdl
from abcdl import constants


def test_constants():
    assert constants.FPS == 30
    assert constants.TICK_NS == 33333333
    assert constants.TIMESCALE == 15360
    assert constants.TICKS_PER_FRAME == 512
    assert constants.TIMESCALE // constants.FPS == constants.TICKS_PER_FRAME
