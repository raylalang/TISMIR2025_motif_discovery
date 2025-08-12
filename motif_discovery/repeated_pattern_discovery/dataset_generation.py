import random
from vector import Vector


def create_data_file_with_random_patterns(n, min_pattern_size, max_pattern_size, max_reps, dimensionality, filename):
    """ Create a dataset of random patterns. """

    f = open(filename, mode='w')

    dataset_vectors = set()

    while len(dataset_vectors) <= n:
        pattern_size = random.randint(min_pattern_size, max_pattern_size)
        pattern = get_random_pattern(pattern_size, dimensionality)

        for v in pattern:
            dataset_vectors.add(v)

        repetitions = random.randint(1, max_reps)
        for _ in range(0, repetitions):
            rand_vector = get_random_vector(dimensionality, 0, 1000)

            for p in pattern:
                translated_vec = p + rand_vector
                dataset_vectors.add(translated_vec)

    for vec in list(dataset_vectors)[0:n]:
        f.write(vector_to_csv_line(vec))


def get_random_pattern(pattern_size, dimensionality):
    pattern = []

    for _ in range(pattern_size):
        pattern.append(get_random_vector(dimensionality, 0, 1000))

    return pattern


def get_random_vector(dimensionality, min_val, max_val):
    components = []
    for _ in range(0, dimensionality):
        random_component = float(random.randint(min_val, max_val))
        random_component += random.randint(1, 4) / 4
        components.append(random_component)

    return Vector(components)


def vector_to_csv_line(vec):
    s = ''
    for i in range(vec.dimensionality()):
        s += str(vec[i]) + ', '

    return s[0:len(s) - 2] + '\n'


def create_random_datafiles(limits, increment, dimensionality):
    for n in range(limits[0], limits[1] + 1, increment):
        create_data_file_with_random_patterns(n, int(n/100), int(n/4), 30, dimensionality,
                                              'testfiles/random_patterns/rand_patterns_' + str(n) + '.csv')


def create_mtp_count_minimizing_dataset(filename, n):
    f = open(filename, mode='w')
    for x in range(0, n):
        f.write(vector_to_csv_line(Vector([x, 1])))


def create_mtp_count_maximizing_dataset(filename, n):
    f = open(filename, mode='w')
    y_incr = 0.01
    y = 0
    for x in range(0, n):
        y += x * y_incr
        f.write(vector_to_csv_line(Vector([x, y])))


def main():
    pass


if __name__ == '__main__':
    main()
