import unittest
from tec import TEC
from vector import Vector
from dataset import Dataset
import heuristics


class HeuristicTest(unittest.TestCase):

    def test_bounding_box(self):
        dataset = Dataset('unittest_data/heuristics_test.csv')
        dataset = Dataset.sort_ascending(dataset)
        tec = TEC([Vector([1, 3]), Vector([3, 5]), Vector([4, 1]), Vector([5, 3])], [0, 3, 4, 6], [Vector([0, 0])])
        self.assertEqual(heuristics.bounding_box_compactness(tec, dataset), 4/9)

    def test_pattern_width(self):
        tec = TEC([Vector([1, 3, 4]), Vector([1, 1, 5]), Vector([5, 1, 2])], [0, 1, 2], [Vector([0, 0, 0])])
        self.assertEqual(heuristics.pattern_width(tec), 4)

    def test_pattern_volume(self):
        tec = TEC([Vector([2, -1, 0]), Vector([-1, 2, -1]), Vector([0, 1, 2])], [0, 1, 2], [Vector([0, 0, 0])])
        self.assertEqual(heuristics.pattern_volume(tec), 27)


if __name__ == '__main__':
    unittest.main()