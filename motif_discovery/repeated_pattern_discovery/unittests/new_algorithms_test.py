import unittest
from dataset import Dataset
from vector import Vector
import new_algorithms
import orig_algorithms
import helpers


class NewAlgorithmsTest(unittest.TestCase):

    def test_own_sia(self):
        dataset = Dataset('unittest_data/test_data1.csv')
        result = new_algorithms.siah(dataset)

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
        result = new_algorithms.siah(dataset)
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

    def test_siatech_with_data_from_meredith(self):
        dataset = Dataset('unittest_data/Meredith2002_fig11.csv')
        result = new_algorithms.siatech(dataset)
        exp_res = helpers.get_tecs_for_Meredith2002_fig11()

        print('Result:')
        helpers.print_tecs(result)
        print('\nExpected:')
        helpers.print_tecs(exp_res)
        print('\n')

        self.assertEqual(len(result), len(exp_res))

        # Check that all tecs in best_tecs are valid tecs.
        for tec in result:
            self.assertTrue(helpers.tec_in_list(tec, exp_res))

    def test_siatec_hash(self):
        dataset = Dataset('unittest_data/rand_patterns.csv')
        result = new_algorithms.siatech(dataset)
        expected = orig_algorithms.siatec(dataset)

        self.assertEqual(len(result), len(expected))

        for tec in result:
            self.assertTrue(helpers.tec_in_list(tec, expected))


if __name__ == '__main__':
    unittest.main()