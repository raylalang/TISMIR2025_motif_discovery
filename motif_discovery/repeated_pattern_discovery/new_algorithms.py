from helpers import vec
from dataset import Dataset
from tec import TEC
from pattern import Pattern
import heuristics
from sys import maxsize


""" Contains algorithms that use hashing to improve the runtime of MTP and TEC computation. """


def siah(d):
    """ Computes the MTPs of dataset d.
        Uses a dictionary/map to avoid having to sort the
        set of difference vectors. Runs in O(kn^2) expected time. """

    d = Dataset.sort_ascending(d)
    # Dictionary of difference vector, index list pairs.
    mtp_dict = {}

    # Compute the difference vectors between points and add
    # the starting indexes to the lists corresponding to the
    # difference vector.
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            diff = d[j] - d[i]
            if diff in mtp_dict:
                mtp_dict[diff].append(i)
            else:
                mtp_dict[diff] = [i]

    # Collect the MTPs from the map.
    mtps = []
    for diff in mtp_dict:
        points = []
        for index in mtp_dict[diff]:
            points.append(d[index])

        mtps.append((diff, points))

    return mtps


def siatech(d):
    d = Dataset.sort_ascending(d)
    # Map of difference vector, index list pairs.
    mtp_map = {}

    # Compute the difference vectors between points and add both
    # the starting and ending index as pair to the lists corresponding to the
    # difference vector.
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            diff = d[j] - d[i]
            if diff in mtp_map:
                mtp_map[diff].append((i, j))
            else:
                mtp_map[diff] = [(i, j)]

    tecs = []
    handled_patterns = set()

    for diff_vec in mtp_map:
        pattern = []
        pattern_indices = []
        mtp = mtp_map[diff_vec]

        for index_pair in mtp:
            pattern_indices.append(index_pair[0])
            pattern.append(d[index_pair[0]])

        vectorized_pattern = Pattern(vec(pattern))

        if vectorized_pattern not in handled_patterns:
            translators = []
            if len(pattern) == 1:
                for point in d:
                    translators.append(point - pattern[0])
            else:
                translators = find_translators_h(pattern, vectorized_pattern, mtp_map, d)

            tecs.append(TEC(pattern, pattern_indices, translators))
            handled_patterns.add(vectorized_pattern)

    return tecs


def siatech_pf(d, c_min):
    """ SIATECH that only returns TECs with compression ratio of at least c_min.
        Computes all TECs and filters out those with insufficient compression ratio (postfiltering). """

    d = Dataset.sort_ascending(d)
    # Map of difference vector, index list pairs.
    mtp_map = {}

    # Compute the difference vectors between points and add both
    # the starting and ending index as pair to the lists corresponding to the
    # difference vector.
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            diff = d[j] - d[i]
            if diff in mtp_map:
                mtp_map[diff].append((i, j))
            else:
                mtp_map[diff] = [(i, j)]

    tecs = []
    handled_patterns = set()

    for diff_vec in mtp_map:
        pattern = []
        pattern_indices = []
        mtp = mtp_map[diff_vec]

        for index_pair in mtp:
            pattern_indices.append(index_pair[0])
            pattern.append(d[index_pair[0]])

        vectorized_pattern = Pattern(vec(pattern))

        if vectorized_pattern not in handled_patterns:
            translators = []
            if len(pattern) == 1:
                for point in d:
                    translators.append(point - pattern[0])
            else:
                translators = find_translators_h(pattern, vectorized_pattern, mtp_map, d)

            tec = TEC(pattern, pattern_indices, translators)
            if heuristics.compression_ratio(tec) >= c_min:
                tecs.append(tec)

            handled_patterns.add(vectorized_pattern)

    return tecs


def find_translators_h(mtp, vectorized_mtp, mtp_map, sorted_dataset):
    target_indices = []
    for index_pair in mtp_map[vectorized_mtp[0]]:
        target_indices.append(index_pair[1])

    for i in range(1, len(vectorized_mtp)):
        v = vectorized_mtp[i]
        index_pair_list = mtp_map[v]

        tmp_target_indices = []
        j = 0
        k = 0
        while j < len(target_indices) and k < len(index_pair_list):
            if target_indices[j] == index_pair_list[k][0]:
                tmp_target_indices.append(index_pair_list[k][1])
                j += 1
                k += 1
            elif target_indices[j] < index_pair_list[k][0]:
                j += 1
            elif target_indices[j] > index_pair_list[k][0]:
                k += 1

        target_indices = tmp_target_indices

    translators = []
    last_point = mtp[len(mtp) - 1]

    for index in target_indices:
        p = sorted_dataset[index]
        translators.append(p - last_point)

    return translators


def siatechf(d, min_cr):
    """ SIATECH that only returns TECs that have compression ratio of at least min_cr. """
    d = Dataset.sort_ascending(d)
    # Map of difference vector, index list pairs.
    mtp_map = {}

    # Compute the difference vectors between points and add both
    # the starting and ending index as pair to the lists corresponding to the
    # difference vector.
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            diff = d[j] - d[i]
            if diff in mtp_map:
                mtp_map[diff].append((i, j))
            else:
                mtp_map[diff] = [(i, j)]

    tecs = []
    handled_patterns = set()

    for diff_vec in mtp_map:
        pattern = []
        pattern_indices = []
        mtp = mtp_map[diff_vec]

        for index_pair in mtp:
            pattern_indices.append(index_pair[0])
            pattern.append(d[index_pair[0]])

        vectorized_pattern = Pattern(vec(pattern))

        if vectorized_pattern not in handled_patterns:
            if cr_upper_bound(pattern, mtp_map, d) >= min_cr:
                translators = []
                if len(pattern) == 1:
                    for point in d:
                        translators.append(point - pattern[0])
                else:
                    translators = find_translators_h(pattern, vectorized_pattern, mtp_map, d)

                tec = TEC(pattern, pattern_indices, translators)

                if heuristics.compression_ratio(tec) >= min_cr:
                    tecs.append(tec)

            handled_patterns.add(vectorized_pattern)

    return tecs


def cr_upper_bound(pattern, mtp_map, dataset):
    """" Computes an upper bound on the compression ratio of pattern. """

    vec_pat = vec(pattern)
    if len(pattern) > 1:
        vec_pat.append(pattern[len(pattern) - 1] - pattern[0])
    min_occ = maxsize
    for v in vec_pat:
        if len(mtp_map[v]) < min_occ:
            min_occ = len(mtp_map[v])

    transl_bound = min(min_occ, len(dataset) - len(pattern) + 1)

    return len(pattern)*transl_bound / (len(pattern) + transl_bound - 1)