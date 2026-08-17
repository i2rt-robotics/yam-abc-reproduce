import numpy as np

from olmo.data.video_loader import TimeSampler


def test_time_sampling_basic():
    ts = TimeSampler(max_frames=4, frame_sample_mode="uniform_last_frame")
    assert np.allclose(ts(2)[1], [0, 2/3, 4/3, 2])
    assert np.allclose(ts(2.1)[1], [0, 2.1/3, 2*2.1/3, 2.1])


def test_time_sampling_max_1fps():
    ts = TimeSampler(max_frames=4, frame_sample_mode="uniform_last_frame", max_fps=1)
    assert np.allclose(ts(0.1)[1], [0, 0.1])
    assert np.allclose(ts(1.1)[1], [0, 1, 1.1])
    assert np.allclose(ts(1.9)[1], [0, 1, 1.9])
    assert np.allclose(ts(2)[1], [0, 1, 2])
    assert np.allclose(ts(2.001)[1], [0, 1, 2, 2.001])
    assert np.allclose(ts(2.1)[1], [0, 1, 2, 2.1])
    assert np.allclose(ts(2.999)[1], [0, 1, 2, 2.999])
    assert np.allclose(ts(3)[1], [0, 1, 2, 3])
    assert np.allclose(ts(3.001)[1], [0, 3.001/3, 2*3.001/3, 3.001])
    assert np.allclose(ts(6.1)[1], [0, 6.1/3, 2*6.1/3, 6.1])
    assert np.allclose(ts(10)[1], [0, 10/3, 2*10/3, 10])


def test_time_sampling_max_2fps():
    ts = TimeSampler(max_frames=5, frame_sample_mode="uniform_last_frame", max_fps=2)
    assert np.allclose(ts(0.1)[1], [0, 0.1])
    assert np.allclose(ts(1.5)[1], [0, 0.5, 1, 1.5])
