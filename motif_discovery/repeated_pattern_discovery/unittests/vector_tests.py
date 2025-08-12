import unittest
from vector import Vector


class VectorTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_dimensionality(self):
        a = Vector([1, 2])
        self.assertEqual(a.dimensionality(), 2)

    def test_get_component(self):
        a = Vector([1, 2, 3])
        self.assertEqual(a[0], 1)
        self.assertEqual(a[1], 2)
        self.assertEqual(a[2], 3)

    def test_eq(self):
        a = Vector([1, 2])
        b = Vector([1, 2])
        self.assertTrue(a == b)

        c = Vector([1, 2.1])
        self.assertFalse(a == c)

    def test_cmp(self):
        a = Vector([1, 2, 3])
        b = Vector([1, 2, 3])
        c = Vector([1, 2, 3.1])
        d = Vector([0, 2, 3])

        self.assertEqual(a.__cmp__(b), 0)
        self.assertTrue(a.__cmp__(c) < 0)
        self.assertTrue(a.__cmp__(d) > 0)
        self.assertTrue(c.__cmp__(a) > 0)

    def test_lt(self):
        self.assertTrue(Vector([1, 2]) < Vector([1, 2.1]))
        self.assertFalse(Vector([1, 2]) < Vector([1, 2]))
        self.assertFalse(Vector([1.1, 2]) < Vector([1, 2]))

    def test_le(self):
        self.assertTrue(Vector([1, 2]) <= Vector([1, 2]))
        self.assertFalse(Vector([1, 2.1]) <= Vector([1, 2]))
        self.assertFalse(Vector([1.1, 2]) <= Vector([1, 2]))

    def test_gt(self):
        self.assertTrue(Vector([1, 2.2]) > Vector([1, 2]))
        self.assertFalse(Vector([1, 2]) > Vector([1, 2]))
        self.assertFalse(Vector([1, 2]) > Vector([1.1, 2]))

    def test_ge(self):
        self.assertTrue(Vector([1, 2]) >= Vector([1, 2]))
        self.assertFalse(Vector([1, 2]) >= Vector([1, 2.1]))
        self.assertFalse(Vector([1, 2]) >= Vector([1.1, 2]))

    def test_add(self):
        self.assertEqual(Vector([1, 2]) + Vector([2, 1]), Vector([3, 3]))
        self.assertEqual(Vector([0, 0, 0, 0, 0]) + Vector([1, 1, 1, 1, 1]), Vector([1, 1, 1, 1, 1]))
        self.assertEqual(Vector([0.5, 0.5, 0.5]) + Vector([1, 1, 0.5]), Vector([1.5, 1.5, 1]))

    def test_sub(self):
        self.assertEqual(Vector([1, 2]) - Vector([2, 1]), Vector([-1, 1]))
        self.assertEqual(Vector([0, 0, 0, 0, 0]) - Vector([1, 1, 1, 1, 1]), Vector([-1, -1, -1, -1, -1]))
        self.assertEqual(Vector([0.5, 0.5, 0.5]) - Vector([1, 1, 0.5]), Vector([-0.5, -0.5, 0]))


if __name__ == "__main__":
    unittest.main()
