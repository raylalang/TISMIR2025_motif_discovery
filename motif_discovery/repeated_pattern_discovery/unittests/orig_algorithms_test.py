import unittest
from dataset import Dataset
from vector import Vector
import orig_algorithms
from tec import TEC
import helpers


class OrigAlgorithmsTest(unittest.TestCase):

    def test_sia(self):
        dataset = Dataset('unittest_data/test_data1.csv')
        result = orig_algorithms.sia(dataset)

        exp_res = []
        exp_res.append((Vector([2, 1]), [Vector([1, 2]), Vector([2, 1])]))
        exp_res.append((Vector([1, -1]), [Vector([1, 2]), Vector([3, 3])]))
        exp_res.append((Vector([3, 0]), [Vector([1, 2])]))
        exp_res.append((Vector([1, 2]), [Vector([2, 1])]))

        results_match = helpers.check_result(result, exp_res)
        if not results_match:
            print('Incorrect result')
            helpers.print_mtps(result)

        self.assertTrue(results_match)

    def test_sia_with_data_from_meredith(self):
        dataset = Dataset('unittest_data/Meredith2002_fig11.csv')
        result = orig_algorithms.sia(dataset)
        helpers.print_mtps(result)

        exp_res = []
        exp_res.append((Vector([0, 1]), [Vector([2, 1]), Vector([2, 2])]))
        exp_res.append((Vector([0, 2]), [Vector([1, 1]), Vector([2, 1])]))
        exp_res.append((Vector([1, -2]), [Vector([1, 3])]))
        exp_res.append((Vector([1, -1]), [Vector([1, 3]), Vector([2, 3])]))
        exp_res.append((Vector([1, 0]), [Vector([1, 1]), Vector([1, 3]), Vector([2, 2])]))
        exp_res.append((Vector([1, 1]), [Vector([1, 1]), Vector([2, 1])]))
        exp_res.append((Vector([1, 2]), [Vector([1, 1])]))
        exp_res.append((Vector([2, -1]), [Vector([1, 3])]))
        exp_res.append((Vector([2, 1]), [Vector([1, 1])]))

        results_match = helpers.check_result(result, exp_res)
        if not results_match:
            print('Incorrect result')
            helpers.print_mtps(result)

        self.assertTrue(results_match)

    def test_siatec_with_data_from_meredith(self):
        dataset = Dataset('unittest_data/Meredith2002_fig11.csv')
        result = orig_algorithms.siatec(dataset)

        exp_res = helpers.get_tecs_for_Meredith2002_fig11()
        results_match = helpers.check_result(result, exp_res)
        if not results_match:
            print('Incorrect result')

        self.assertTrue(results_match)

    def test_coverage(self):
        tec = TEC([Vector([2, 1]), Vector([2, 2])], [], [Vector([0, 0]), Vector([1, 1])])
        covered_points = tec.coverage()
        self.assertTrue(Vector([2, 1]) in covered_points)
        self.assertTrue(Vector([2, 2]) in covered_points)
        self.assertTrue(Vector([3, 2]) in covered_points)
        self.assertTrue(Vector([3, 3]) in covered_points)

    def test_cosiatec_with_data_from_meredith2002(self):
        dataset = Dataset('unittest_data/Meredith2002_fig11.csv')
        best_tecs = orig_algorithms.cosiatec(dataset)
        all_tecs = helpers.get_tecs_for_Meredith2002_fig11()

        helpers.print_tecs(all_tecs)
        helpers.print_tecs(best_tecs)

        # Check that all tecs in best_tecs are valid tecs.
        for tec in best_tecs:
            self.assertTrue(helpers.tec_in_list(tec, all_tecs))

        # Check coverage of best_tecs.
        covered_points = []

        for tec in best_tecs:
            cov = tec.coverage()
            for p in cov:
                covered_points.append(p)

        for i in range(0, len(dataset)):
            self.assertTrue(dataset[i] in covered_points)

    def test_cosiatec_with_random_data(self):
        dataset = Dataset('unittest_data/rand_patterns.csv')
        all_tecs = orig_algorithms.siatec(dataset)
        helpers.print_tecs(all_tecs)

        best_tecs = orig_algorithms.cosiatec(dataset)
        helpers.print_tecs(best_tecs)

        # Check that all tecs in best_tecs are valid tecs.
        for tec in best_tecs:
            self.assertTrue(helpers.tec_in_list(tec, all_tecs))

        # Check coverage of best_tecs.
        covered_points = []

        for tec in best_tecs:
            cov = tec.coverage()
            for p in cov:
                covered_points.append(p)

        for i in range(0, len(dataset)):
            self.assertTrue(dataset[i] in covered_points)

    def test_siatec_compress_with_data_from_meredith2002(self):
        dataset = Dataset('unittest_data/Meredith2002_fig11.csv')
        best_tecs = orig_algorithms.siatec_compress(dataset)
        all_tecs = helpers.get_tecs_for_Meredith2002_fig11()

        helpers.print_tecs(all_tecs)
        helpers.print_tecs(best_tecs)

        # Check that all tecs in best_tecs are valid tecs.
        for tec in best_tecs:
            self.assertTrue(helpers.tec_in_list(tec, all_tecs))

        # Check coverage of best_tecs.
        covered_points = []

        for tec in best_tecs:
            cov = tec.coverage()
            for p in cov:
                covered_points.append(p)

        for i in range(0, len(dataset)):
            self.assertTrue(dataset[i] in covered_points)

    def test_siatec_compress_with_random_data(self):
        dataset = Dataset('unittest_data/rand_patterns.csv')
        all_tecs = orig_algorithms.siatec(dataset)
        helpers.print_tecs(all_tecs)

        best_tecs = orig_algorithms.siatec_compress(dataset)
        helpers.print_tecs(best_tecs)

        # Check that all tecs in best_tecs are valid tecs.
        for tec in best_tecs:
            self.assertTrue(helpers.tec_in_list(tec, all_tecs))

        # Check coverage of best_tecs.
        covered_points = []

        for tec in best_tecs:
            cov = tec.coverage()
            for p in cov:
                covered_points.append(p)

        for i in range(0, len(dataset)):
            self.assertTrue(dataset[i] in covered_points)

    def test_vec(self):
        pattern = [Vector([0, 0]), Vector([0, 2]), Vector([1, 1])]
        vec_pattern = orig_algorithms.vec(pattern)
        self.assertEqual(vec_pattern, [Vector([0, 2]), Vector([1, -1])])


if __name__ == '__main__':
    unittest.main()