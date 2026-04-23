import numpy as np

from ood_detector import OODDetector


def test_leave_one_out_nearest_distances_matches_naive():
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )

    expected = np.zeros(len(features), dtype=np.float32)
    for idx in range(len(features)):
        diffs = features - features[idx : idx + 1]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[idx] = np.inf
        expected[idx] = dists.min()

    actual = OODDetector._leave_one_out_nearest_distances(features)

    assert np.allclose(actual, expected)
