import numpy as np

from src.stats import Stats


class TestStats:

    def test_stats_single_value(self):
        stats = Stats()
        values = [1, 2, 3, 4, 5]
        for x in values:
            stats.add_to_stats(x)
        assert stats.count == 5
        assert stats.mean == 3
        assert np.isclose(stats.std, 1.41421356)
        assert stats.min == 1
        assert stats.max == 5

    def test_stats_multi_features(self):
        stats = Stats()
        values = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        for x in values:
            stats.add_to_stats(x)
        assert stats.count == 5
        assert (stats.mean == [3, 4]).all()
        assert np.allclose(stats.std, [1.41421356, 1.41421356])
        assert (stats.min == [1, 2]).all()
        assert (stats.max == [5, 6]).all()

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

    def test_3d(self):
        stats = Stats()
        values = [[[1], [2]], [[2], [3]], [[3], [4]], [[4], [5]], [[5], [6]]]
        stats.add_to_stats(values)
        assert stats.count == 5
        assert (stats.mean == [[3], [4]]).all()
        assert np.allclose(stats.std, [[1.41421356], [1.41421356]])
        assert (stats.min == [[1], [2]]).all()
        assert (stats.max == [[5], [6]]).all()
