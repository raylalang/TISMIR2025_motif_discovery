from random_multipliers import RandMult


class Vector:
    """ Defines a vector and the operations required for expressing symbolic music
        data as vectors as defined in [Meredith2002] """

    __components = []
    __hash_value = None

    def __init__(self, components):
        self.__components = components

    def dimensionality(self):
        """ Returns the dimensionality of the vector. """
        return len(self.__components)

    def __getitem__(self, item):
        """ Returns the ith component of the vector """
        return self.__components[item]

    def __cmp__(self, other):
        """ Vectors are compared using lexicographical ordering. """

        for i in range(0, self.dimensionality()):
            if self[i] < other[i]:
                return -1
            if self[i] > other[i]:
                return 1

        return 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def __str__(self):
        string_repr = '('
        for i in range(0, self.dimensionality()):
            string_repr += str(self[i]) + ', '

        string_repr = string_repr[0: len(string_repr)-2] + ')'
        return string_repr

    def __add__(self, other):
        summed_components = []

        for i in range(0, self.dimensionality()):
            summed_components.append(self[i] + other[i])

        return Vector(summed_components)

    def __sub__(self, other):
        subs_components = []

        for i in range(0, self.dimensionality()):
            subs_components.append(self[i] - other[i])

        return Vector(subs_components)

    def is_zero(self):
        for component in self.__components:
            if component != 0:
                return False

        return True

    def __hash__(self):
        """ A hash function based on theorem 3.1 of [Lemire2014], with K = 64, L = 32.
            The components are handled as 64 bit floats and not 32 bit integers, so
            this does not necessarily ensure strongly universal hashing. """

        if not self.__hash_value:
            s = RandMult().multiplier(0)
            i = 1
            for d in self.__components:
                s += RandMult().multiplier(i) * d
                i += 1

            self.__hash_value = int(s % 2**64 / 2**31)

        return self.__hash_value

    @staticmethod
    def vector_set_to_str(vector_set):
        if not vector_set:
            return '{}'

        vset_str = '{'
        for v in vector_set:
            vset_str += str(v) + ', '

        return vset_str[0:len(vset_str) - 2] + '}'

    @staticmethod
    def zero_vector(dimensionality):
        return Vector([0 for _ in range(dimensionality)])

