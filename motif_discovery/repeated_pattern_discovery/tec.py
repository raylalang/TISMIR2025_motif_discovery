from vector import Vector


class TEC:
    """ Class for Translational Equivalence Class [Meredith2002]. """

    _pattern = []
    _pattern_indices = []
    _translators = []
    _coverage = None

    def __init__(self, pattern, pattern_indices, translators):
        self._pattern = pattern
        self._pattern_indices = pattern_indices
        self._pattern_indices.sort()
        self._translators = translators

    def get_pattern(self):
        return self._pattern

    def get_pattern_indices(self):
        return self._pattern_indices

    def get_translators(self):
        return self._translators

    def __lt__(self, other):
        if len(self._pattern) < len(other.get_pattern()):
            return True
        if len(self._pattern) > len(other.get_pattern()):
            return False

        return self._pattern < other.get_pattern()

    def __str__(self):
        self._pattern.sort()
        self._translators.sort()
        return Vector.vector_set_to_str(self._pattern) + ', ' + Vector.vector_set_to_str(self._translators)

    def __eq__(self, other):
        if len(self.get_pattern()) != len(other.get_pattern()):
            return False

        if len(self.get_translators()) != len(other.get_translators()):
            return False

        return self.coverage() == other.coverage()

    def get_bounding_box(self):
        """ Returns the vectors that limit the minimum axis aligned bounding box of
            the pattern of the TEC.
            Returns two vectors: min_vector, max_vector """

        dim = self.get_pattern()[0].dimensionality()
        max_components = []
        min_components = []
        for _ in range(0, dim):
            max_components.append(float('-inf'))
            min_components.append(float('inf'))

        for p in self.get_pattern():
            for i in range(0, dim):
                component = p[i]
                if component > max_components[i]:
                    max_components[i] = component

                if component < min_components[i]:
                    min_components[i] = component

        return Vector(min_components), Vector(max_components)

    def coverage(self):
        """ Computes the set of points covered by the TEC as defined in eq. 4 of [Meredith2013].
            Returns the covered points as a set. """

        if self._coverage:
            return self._coverage

        covered_points = set()
        pattern = self.get_pattern()

        translators_contains_zero_vector = False

        for translator in self.get_translators():
            if translator.is_zero():
                translators_contains_zero_vector = True

            for point in pattern:
                covered_points.add(point + translator)

        if not translators_contains_zero_vector:
            for point in pattern:
                covered_points.add(point)

        self._coverage = covered_points
        return self._coverage
