from random import randint


class RandMult:
    """ Singleton class for providing random integers in the range [0, 2^64) for hashing. """
    class __RandMult:
        _multipliers = []

        def __init__(self):
            self._multipliers = self.generate_random_numbers_for_hashing(1000)

        def multiplier(self, i):
            while len(self._multipliers) < i + 1:
                self._multipliers.extend(self.generate_random_numbers_for_hashing(10))

            return self._multipliers[i]

        def generate_random_numbers_for_hashing(self, n):
            """ This is used for generating the random multipliers in range [0, 2^64) that are used for hashing. """
            rand_vec = []
            for _ in range(n):
                rand_vec.append(randint(0, 2 ** 64))
            return rand_vec

    instance = None

    def __init__(self):
        if not RandMult.instance:
            RandMult.instance = RandMult.__RandMult()

    def multiplier(self, i):
        return RandMult.instance.multiplier(i)


