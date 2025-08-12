from copy import deepcopy
from random_multipliers import RandMult


class Pattern:
    """ A simple container class for patterns so that they can be hashed. """

    _points = []
    _hash = None

    def __init__(self, points):
        self._points = deepcopy(points)

    def __getitem__(self, item):
        return self._points[item]

    def __eq__(self, other):
        if self._points == other._points:
            return True

        return False

    def __len__(self):
        return len(self._points)

    def __hash__(self):
        """ A hash function based on theorem 3.1 of [Lemire2014], with K = 64, L = 32.
            The components are handled as 64 bit floats and not 32 bit integers, so
            this does not necessarily ensure strongly universal hashing. """

        if not self._hash:
            mult_ind = 0
            m = RandMult()
            sum_val = m.multiplier(mult_ind)
            for vec in self._points:
                for i in range(vec.dimensionality()):
                    mult_ind += 1
                    sum_val += m.multiplier(mult_ind) * vec[i]

            self._hash = int(sum_val % 2 ** 64 / 2 ** 31)

        return self._hash
