import numpy as np

from src.stats import Stats


class TestStats:

    def test_stats_single_value(self):
        stats = Stats()
        values = [-2, -1, 0, 1, 2]
        for x in values:
            stats.add_to_stats(x)
        assert stats.count == 5
        assert stats.mean == 0
        assert np.isclose(stats.std, 1.41421356)
        assert stats.min == -2
        assert stats.max == 2
        assert (stats.samples == [-2, 1]).all()

    def test_stats_multi_features(self):
        stats = Stats()
        values = [[1, 2], [2, 3], [3, 4], [5, 4], [6, 5]]
        for x in values:
            stats.add_to_stats(x)
        assert stats.count == 5
        assert (stats.mean == [3.4, 3.6]).all()
        assert np.allclose(stats.std, [1.8547237, 1.0198039])
        assert (stats.min == [1, 2]).all()
        assert (stats.max == [6, 5]).all()
        assert (stats.samples == [[1, 2], [5, 4]]).all()

    def test_stats_batched(self):
        stats = Stats()
        values = [[[1], [2]], [[2], [3]], [[3], [4]], [[4], [5]], [[5], [6]]]
        for x in values:
            stats.add_to_stats(x)
        assert stats.count == 10
        assert stats.mean == 3.5
        assert np.allclose(stats.std, 1.5)
        assert stats.min == 1
        assert stats.max == 6
        assert stats.samples == [1]

    def test_3d(self):
        stats = Stats()
        values = [[[[1], [2]], [[2], [3]]], [[[3], [4]], [[5], [4]], [[6], [5]]]]
        for x in values:
            stats.add_to_stats(x)
        assert stats.count == 5
        assert (stats.mean == [[3.4], [3.6]]).all()
        assert np.allclose(stats.std, [[1.8547237], [1.0198039]])
        assert (stats.min == [[1], [2]]).all()
        assert (stats.max == [[6], [5]]).all()
        assert (stats.samples == [[1, 2], [5, 4]]).all()
