from vector import Vector
from functools import cmp_to_key
from copy import deepcopy
from operator import itemgetter
from dataset import Dataset
import heuristics
from tec import TEC
from new_algorithms import siatech
from helpers import vec

""" Contains implementations of algorithms based on SIA [Meredith2002]. """


def sia(d):
    """ Implements SIA as described in [Meredith2002]

        Returns the set of MTPs as a list of tuples where the
        first element is the translation vector and the second
        element is a list of vectors that form the corresponding pattern. """

    # Step 1: Sort the dataset in ascending order
    d = Dataset.sort_ascending(d)

    # Step 2: Compute the vector table V
    v = []

    for i in range(0, len(d)):
        for j in range(i + 1, len(d)):
            v.append((d[j] - d[i], i))

    # Step 3: Sort v
    v.sort()

    # Step 4: Compute MTPs from v
    mtps = compute_mtps(v, d)

    return mtps


def compute_mtps(v, d):
    """ This implements the function in [Meredith2002] Fig. 18.
        Instead of printing out the MTPs they are returned as a list of doubles. """

    mtps = []

    i = 0

    while i < len(v):
        pattern = []
        pattern.append(d[v[i][1]])

        j = i + 1
        while j < len(v) and v[j][0] == v[i][0]:
            pattern.append(d[v[j][1]])
            j += 1

        mtps.append((v[i][0], pattern))
        i = j

    return mtps


def siatec(d):
    """ Implements the SIATEC algorithm described in [Meredith2002].

        Returns the TECs of d as a list of pairs, where the first
        element is the MTP and the second element is the list of translators. """

    # Step 1: Sort dataset
    d = Dataset.sort_ascending(d)

    # Step 2 and 3: Compute vector tables w and v
    w = []
    v = []
    for i in range(0, len(d)):
        w.append([])
        for j in range(0, len(d)):
            w[i].append((d[j] - d[i], i))
            if j > i:
                v.append(w[i][j])

    # Step 4: Sort v
    v.sort()

    # Step 5: Vectorize the MTPs.
    x = compute_x(d, v)

    # Step 6: Sort x
    x.sort(key=cmp_to_key(x_cmp))

    # Step 7: Compute TECs from sorted x.
    tecs = compute_tecs(x, v, w, d)

    return tecs


def compute_x(d, v):
    """ Computes the set X as defined in [Meredith2002] Fig. 20. """

    i = 0
    x = []

    while i < len(v):
        q = []
        j = i + 1

        while j < len(v) and v[j][0] == v[i][0]:
            q.append(d[v[j][1]] - d[v[j - 1][1]])
            j += 1

        x.append((i, q))
        i = j

    return x


def x_cmp(x1, x2):
    """ Implements the comparison of elements of set x as defined
        in [Meredith2002] SIATEC step 6. """

    q1 = x1[1]
    q2 = x2[1]

    # Compare length of patterns first.
    if len(q1) < len(q2):
        return -1
    if len(q1) > len(q2):
        return 1

    # Compare patterns lexicographically.
    if q1 < q2:
        return -1
    if q1 > q2:
        return 1

    # If patterns are equal, then compare indices.
    return x1[0] - x2[0]


def compute_tecs(y, v, w, d):
    """ Implements algorithm of fig. 23 in [Meredith2002]. """

    i = 0
    tecs = []

    while i < len(y):
        j = y[i][0]
        pattern_indices = []
        while j < len(v) and v[j][0] == v[y[i][0]][0]:
            pattern_indices.append(v[j][1])
            j += 1

        pattern = collect_pattern(pattern_indices, d)
        translators = find_translators(pattern_indices, w, len(d))
        tecs.append(TEC(pattern, pattern_indices, translators))

        i += 1
        while i < len(y) and y[i][1] == y[i - 1][1]:
            i += 1

    return tecs


def collect_pattern(pattern_indices, d):
    """ Algorithm in Fig. 24 of [Meredith2002] but instead
        of printing the pattern is collected into a list. """

    pattern = []
    for i in pattern_indices:
        pattern.append(d[i])
    return pattern


def find_translators(pattern_indices, w, data_size):
    """ Implements the algorithm in Fig. 25 of [Meredith2002] """
    translators = []

    pattern_len = len(pattern_indices)

    # The case of a pattern with size one needs to be handled separately.
    if pattern_len == 1:
        for j in range(0, len(w[0])):
            translators.append(w[pattern_indices[0]][j][0])
        return translators

    # This list keeps track of the in column indices (=rows) for columns in the difference
    # vector table. in_col_ind[k] is the row number for the column k.
    in_col_ind = []
    for _ in range(0, pattern_len):
        in_col_ind.append(0)

    finished = False
    k = 1

    while not finished:
        if in_col_ind[k] <= in_col_ind[k - 1]:
            in_col_ind[k] = in_col_ind[k - 1] + 1

        while in_col_ind[k] <= data_size - pattern_len + k \
                and w[pattern_indices[k]][in_col_ind[k]][0] < w[pattern_indices[k - 1]][in_col_ind[k - 1]][0]:
            in_col_ind[k] += 1

        if in_col_ind[k] > data_size - pattern_len + k:
            finished = True
        elif w[pattern_indices[k]][in_col_ind[k]][0] > w[pattern_indices[k - 1]][in_col_ind[k - 1]][0]:
            k = 1
            in_col_ind[0] += 1
            if in_col_ind[0] > data_size - pattern_len + 1:
                finished = True
        elif k == len(pattern_indices) - 1:
            translators.append(w[pattern_indices[k]][in_col_ind[k]][0])
            k = 0
            while k < pattern_len:
                in_col_ind[k] += 1
                if in_col_ind[k] > data_size - pattern_len + k:
                    finished = True
                    k = pattern_len - 1
                k += 1
            k = 1
        else:
            k += 1

    return translators


def cosiatec(d):
    """ Implements the COSIATEC algorithm as described in [Meredith2013]. """
    d = Dataset.sort_ascending(d)
    p = deepcopy(d)
    best_tecs = []

    while p:
        best_tec = get_best_tec(p, d)
        best_tecs.append(best_tec)
        p.remove_all(list(best_tec.coverage()))

    return best_tecs


def siatec_compress(d):
    """ Implements the SIATECCompress algorithm described in [Meredith2013].
        Removing redundant translators is not implemented as its implementation is not described in [Meredith2013]. """

    d = Dataset.sort_ascending(d)
    v, w = compute_vector_tables(d)
    mcps = compute_mtp_cis_pairs(v)
    remove_trans_eq_mtps(mcps)
    tecs = compute_tecs_from_mcps(d, w, mcps)
    add_conjugate_tecs(tecs, d)
    sort_tecs_by_quality(tecs, d)
    return compute_encoding(tecs, d)


def remove_trans_eq_mtps(mcps):
    """ Removes translationally equivalent MTPs from mcps by
        removing redundant copies of MTPs that have the same
        vectorized representation. """

    i = 0
    while i < len(mcps):
        mcp = mcps[i]
        vectorized = vec(mcp[0])

        j = i + 1
        while j < len(mcps):
            if vectorized == vec(mcps[j][0]):
                mcps.pop(j)
            else:
                j += 1

        i += 1


def compute_tecs_from_mcps(d, v, mcps):
    tecs = []

    for mcp in mcps:
        tecs.append(get_tec_for_mtp(mcp[1], v, d))

    return tecs


def add_conjugate_tecs(tecs, d):
    conj_tecs = []
    for tec in tecs:
        conj_tecs.append(get_conj(tec, d))

    tecs += conj_tecs


def sort_tecs_by_quality(tecs, sorted_dataset):
    tecs.sort(key=cmp_to_key(lambda tec1, tec2: tec_quality_cmp(tec1, tec2, sorted_dataset)))


def tec_quality_cmp(tec1, tec2, sorted_dataset):
    if is_better_tec(tec1, tec2, sorted_dataset):
        return -1
    else:
        return 1


def compute_encoding(tecs, d):
    """ Implements algorithm in Figure 7 of [Meredith2013]. """

    best_tecs = []
    cover = []

    for i in range(0, len(tecs)):
        tec = tecs[i]
        s = tec.coverage()
        s_diff_cover = deepcopy(s)
        for p in cover:
            if p in s_diff_cover:
                s_diff_cover.remove(p)

        if len(s_diff_cover) > len(tec.get_pattern()) + len(tec.get_translators()) - 1:
            best_tecs.append(tec)
            cover += s

            dataset_covered = True
            for p in d:
                if p not in cover:
                    dataset_covered = False
                    break

            if dataset_covered:
                break

    residual = set(d) - set(cover)
    if residual:
        best_tecs.append(TEC(list(residual), [], []))

    return best_tecs


def get_best_tec(p, d):
    """ Finds the best TEC in p, the copy of the dataset d. Implements algorithm in Figure 2 of [Meredith2013].  """

    v, w = compute_vector_tables(p)
    mcps = compute_mtp_cis_pairs(v)
    best_tec = None

    for i in range(0, len(mcps)):
        mcp = mcps[i]
        mtp_indices = mcp[1]
        tec = get_tec_for_mtp(mtp_indices, w, p)
        conj = get_conj(tec, d)

        if not best_tec or is_better_tec(tec, best_tec, d):
            best_tec = tec
        if is_better_tec(conj, best_tec, d):
            best_tec = conj

    return best_tec


def get_best_tech(p, d):
    """ Finds the best TEC in p by using SIATECH.  """

    tecs = siatech(p)
    best_tec = None

    for tec in tecs:
        if not best_tec or is_better_tec(tec, best_tec, d):
            best_tec = tec

    return best_tec


def compute_vector_tables(p):
    """ Compute the vector table V as defined in [Meredith2013],
        and the vector table W as needed by SIATEC [Meredith2002]. """

    v = []
    w = []

    for i in range(0, len(p)):
        v.append([])
        w.append([])
        for j in range(0, len(p)):
            v[i].append((p[i] - p[j], p[j], j))
            w[i].append((p[j] - p[i], i))

    return v, w


def compute_mtp_cis_pairs(v):
    """ Implements algorithm of Figure 3 from [Meredith2013].
        Returns a list of pairs where the first element is the set of
        vectors in the MTP and the second element is the list of indices
        for those vectors in the sorted dataset. """

    w = []
    for i in range(0, len(v)):
        for j in range(0, len(v[i])):
            w.append(v[i][j])

    w.sort(key=itemgetter(0))

    mtps = []
    ciss = []
    vec = w[0][0]
    mtp = [w[0][1]]
    cis = [w[0][2]]

    for i in range(1, len(w)):
        vpi = w[i]
        if vpi[0] == vec:
            mtp.append(vpi[1])
            cis.append(vpi[2])
        else:
            mtps.append(mtp)
            ciss.append(cis)
            mtp = [vpi[1]]
            cis = [vpi[2]]
            vec = vpi[0]

    mtps.append(mtp)
    ciss.append(cis)

    mcps = []
    for i in range(0, len(mtps)):
        mcps.append((mtps[i], ciss[i]))

    return mcps


def get_tec_for_mtp(pattern_indices, w, p):
    """ Find the TEC for the given MTP. This function is not described in
        [Meredith2013] but is said to use the logic of SIATEC, so this function
        uses the logic of finding translators for the pattern used in SIATEC. """

    translators = find_translators(pattern_indices, w, len(p))
    pattern = []
    for index in pattern_indices:
        pattern.append(p[index])

    tec = TEC(pattern, pattern_indices, translators)
    return tec


def get_conj(tec, sorted_dataset):
    """ Computes the conjugate of a TEC as defined in equations
        7-9 of [Meredith2013]. """

    p_0 = tec.get_pattern()[0]
    conj_tec_pattern_ind = [tec.get_pattern_indices()[0]]
    p_prime = [p_0]

    for translator in tec.get_translators():
        if not translator.is_zero():
            p_prime_vec = p_0 + translator
            p_prime.append(p_prime_vec)

            # Find the index for the point in the pattern of the conjugate TEC.
            if p_prime_vec < p_0:
                for i in range(tec.get_pattern_indices()[0], -1, -1):
                    if sorted_dataset[i] == p_prime_vec:
                        conj_tec_pattern_ind.append(i)
                        break
            else:
                for i in range(tec.get_pattern_indices()[0], len(sorted_dataset)):
                    if sorted_dataset[i] == p_prime_vec:
                        conj_tec_pattern_ind.append(i)
                        break

    v_prime = [Vector.zero_vector(p_0.dimensionality())]
    for point in tec.get_pattern():
        p = point - p_0
        if not p.is_zero():
            v_prime.append(point - p_0)

    conj_tec = TEC(p_prime, conj_tec_pattern_ind, v_prime)

    return conj_tec


def is_better_tec(tec1, tec2, sorted_dataset):
    """ Implements algorithm from Figure 5 of [Meredith2013].
        Added else ifs so that the algorithm works correctly as described in the article text. """

    if heuristics.compression_ratio(tec1) > heuristics.compression_ratio(tec2):
        return True
    elif heuristics.compression_ratio(tec1) < heuristics.compression_ratio(tec2):
        return False

    if heuristics.bounding_box_compactness(tec1, sorted_dataset) > heuristics.bounding_box_compactness(tec2, sorted_dataset):
        return True
    elif heuristics.bounding_box_compactness(tec1, sorted_dataset) < heuristics.bounding_box_compactness(tec2,
                                                                                                         sorted_dataset):
        return False

    if len(tec1.coverage()) > len(tec2.coverage()):
        return True
    elif len(tec1.coverage()) < len(tec2.coverage()):
        return False

    # Compare pattern sizes
    if len(tec1.get_pattern()) > len(tec2.get_pattern()):
        return True
    elif len(tec1.get_pattern()) < len(tec2.get_pattern()):
        return False

    if heuristics.pattern_width(tec1) < heuristics.pattern_width(tec2):
        return True
    elif heuristics.pattern_width(tec1) > heuristics.pattern_width(tec2):
        return False

    if heuristics.pattern_volume(tec1) < heuristics.pattern_volume(tec2):
        return True

    return False


def siact(d, compactness_th, cardinality_th, mtp_algorithm=sia):
    """ Implements the SIACT algorithm defined in Definition 7.2 of [Collins2011].
        Takes as its parameter the mtp_algorithm used for finding the MTPs of dataset d. """

    # First find MTPs of the dataset.
    mtps = mtp_algorithm(d)

    # Compactness trawl through each MTP
    compacted_mtps = []

    for mtp in mtps:
        compacted_mtps += compactness_trawl(mtp, compactness_th, cardinality_th, d)

    return compacted_mtps


def compactness_trawl(mtp, compactness_th, cardinality_th, sorted_dataset):
    """ Performs compactness trawling on an MTP as defined in Definition 7.2 of [Collins2011].
        Returns a list of patterns obtained from the input mtp.
        compactness_th is the compactness and cardinality_th the cardinality threshold. """

    diff_vector = mtp[0]
    pattern = mtp[1]

    compacted_mtps = []
    i = 0
    j = 0
    whole_pattern_compact = True

    while j < len(pattern) - 1:
        subpattern = pattern[i:j + 2]
        begin, end = heuristics.find_pattern_indices(subpattern, sorted_dataset)

        compactness = heuristics.compactness(begin, end, len(subpattern), sorted_dataset)
        if compactness < compactness_th:
            whole_pattern_compact = False
            compact_pattern = pattern[i:j + 1]

            if len(compact_pattern) >= cardinality_th:
                compacted_mtps.append((diff_vector, compact_pattern))

            i = j + 1

        j += 1

    # If the whole pattern has compactness above a, then check to see if it is large enough.
    if whole_pattern_compact and len(pattern) >= cardinality_th:
        compacted_mtps.append(mtp)

    return compacted_mtps


def siar(d, r):
    """ Implements SIAR as defined in [Collins2011] and [Meredith2016] Fig.13.14.
        Takes as parameters the dataset d and the number of subdiagonals r."""

    d = Dataset.sort_ascending(d)

    # Compute r subdiagonals of vector table and store in diffs_on_diagonals
    diffs_on_diagonals = []
    for i in range(0, len(d) - 1):
        j = i + 1
        while j < len(d) and j <= i + r:
            diffs_on_diagonals.append((d[j] - d[i], i))
            j += 1

    # Store patterns in transl_patterns by sorting and segmenting diffs_on_diagonals
    diffs_on_diagonals.sort()
    transl_patterns = []
    diff_vec = diffs_on_diagonals[0][0]
    pattern_points = [d[diffs_on_diagonals[0][1]]]
    for i in range(1, len(diffs_on_diagonals)):
        if diffs_on_diagonals[i][0] == diff_vec:
            pattern_points.append(d[diffs_on_diagonals[i][1]])
        else:
            transl_patterns.append(pattern_points)
            pattern_points = [d[diffs_on_diagonals[i][1]]]
            diff_vec = diffs_on_diagonals[i][0]

    transl_patterns.append(pattern_points)

    # For each pattern in transl_patterns, find difference vectors between pattern points
    # and store in within_pattern_diffs
    within_pattern_diffs = []
    for i in range(0, len(transl_patterns)):
        pattern_points = transl_patterns[i]
        for j in range(0, len(pattern_points) - 1):
            for k in range(j + 1, len(pattern_points)):
                within_pattern_diffs.append(pattern_points[k] - pattern_points[j])

    # Remove duplicates from within_pattern_diffs and order vectors by decreasing frequency.
    within_pattern_diffs.sort()
    within_patt_diffs_sorted_by_freq = []

    if within_pattern_diffs:
        diff_vec = within_pattern_diffs[0]
        freq = 1
        for i in range(1, len(within_pattern_diffs)):
            if within_pattern_diffs[i] == diff_vec:
                freq += 1
            else:
                within_patt_diffs_sorted_by_freq.append((diff_vec, freq))
                freq = 1
                diff_vec = within_pattern_diffs[i]

        within_patt_diffs_sorted_by_freq.append((diff_vec, freq))
        within_patt_diffs_sorted_by_freq.sort(key=itemgetter(1), reverse=True)

    # Find the MTP for each vector in within_patt_diffs_sorted_by_freq, store it in mtps and return mtps
    mtps = []
    for i in range(0, len(within_patt_diffs_sorted_by_freq)):
        mtps.append(find_mtp(d, within_patt_diffs_sorted_by_freq[i][0]))

    return mtps


def find_mtp(d, diff_vec):
    """ Find the MTP for diff_vec by finding the intersection of the sorted dataset
        d and the dataset translated by -diff_vec. """

    translated_d = []

    for point in d:
        translated_d.append(point - diff_vec)

    i_d = 0
    i_td = 0
    intersection = []
    while i_d < len(d) and i_td < len(translated_d):
        if d[i_d] == translated_d[i_td]:
            intersection.append(d[i_d])
            i_d += 1
            i_td += 1
        elif d[i_d] < translated_d[i_td]:
            i_d += 1
        elif d[i_d] > translated_d[i_td]:
            i_td += 1

    mtp = (diff_vec, intersection)
    return mtp


def forths_algorithm(d, c_min, sigma_min):
    """ Implements Forth's algorithm [Forth2012], [Meredith2016].
        Returns two lists of TECs, primary and secondary TECs. """

    tecs = siatec(d)
    tec_saliences = compute_tec_saliences(tecs, d)

    return forth_cover(d, tecs, tec_saliences, c_min, sigma_min)


def compute_tec_saliences(tecs, d):
    """ Computes the saliences for TECs as defined in equations. 3.6-3.15 of [Forth2012]. """

    compr_ratios = []
    max_compr_ratio = -1
    min_compr_ratio = len(d)

    compactnesses = []
    max_compactness = -1
    min_compactness = 2

    for tec in tecs:
        compr_ratio = heuristics.compression_ratio(tec)
        if compr_ratio > max_compr_ratio:
            max_compr_ratio = compr_ratio
        if compr_ratio < min_compr_ratio:
            min_compr_ratio = compr_ratio

        compr_ratios.append(compr_ratio)

        compactness = compute_max_compactness(tec, d)
        if compactness > max_compactness:
            max_compactness = compactness
        if compactness < min_compactness:
            min_compactness = compactness

        compactnesses.append(compactness)

    tec_saliences = []

    compactness_denom = (max_compactness - min_compactness)
    if compactness_denom == 0:
        compactness_denom = 1

    compr_ratio_denom = (max_compr_ratio - min_compr_ratio)
    if compr_ratio_denom == 0:
        compr_ratio_denom = 1

    for i in range(len(tecs)):
        # Normalization based on eq. 3.14 of [Forth2012].
        normalized_compactness = (compactnesses[i] - min_compactness) / compactness_denom
        normalized_compr_ratio = (compr_ratios[i] - min_compr_ratio) / compr_ratio_denom

        tec_saliences.append(normalized_compr_ratio * normalized_compactness)

    return tec_saliences


def compute_max_compactness(tec, d):
    """ Returns the maximum compactness of any occurrence of the pattern of the TEC.
        This does not take voice information into consideration as in [Forth2012] eq. 3.13. """

    translators = tec.get_translators()

    zero_vector = Vector.zero_vector(translators[0].dimensionality())
    if zero_vector not in translators:
        translators.append(zero_vector)

    max_comp = 0
    pattern_size = len(tec.get_pattern())

    for translator in tec.get_translators():
        transl_pattern = [v + translator for v in tec.get_pattern()]
        begin, end = heuristics.find_pattern_indices(transl_pattern, d)
        compactness = heuristics.compactness(begin, end, pattern_size, d)
        if compactness > max_comp:
            max_comp = compactness

    return max_comp


def forth_cover(d, tecs, tec_saliences, c_min, sigma_min):
    """ Finds the a cover from the TECs. Based on the pseudocode in [Meredith2016] Fig. 13.10.
        Modified to return the primary and secondary TECs as separate lists instead of returning a list of lists
        of sets (TEC coverages). """

    selected_tecs = []

    points_covered = set()
    found = True

    while not is_dataset_covered(points_covered, d) and found:
        found = False
        max_salience = 0
        best_tec = None
        index_of_best = None
        indexes_of_removable = []
        for i in range(len(tecs)):
            num_new_points_in_cover = len(tecs[i].coverage() - points_covered)
            if num_new_points_in_cover < c_min:
                indexes_of_removable.append(i)
                continue
            weighed_salience = num_new_points_in_cover * tec_saliences[i]
            if weighed_salience > max_salience:
                max_salience = weighed_salience
                best_tec = tecs[i]
                index_of_best = i

        if best_tec:
            indexes_of_removable.append(index_of_best)
            found = True
            points_covered = points_covered | best_tec.coverage()
            i = 0
            primary_found = False
            while not primary_found and i < len(selected_tecs):
                if len((selected_tecs[i][0].coverage() & best_tec.coverage())) / len(selected_tecs[i][0].coverage()) > sigma_min:
                    selected_tecs[i].append(best_tec)
                    primary_found = True
                i += 1

            if not primary_found:
                selected_tecs.append([best_tec])

            remove_from_saliences = []
            remove_from_tecs = []
            for index in indexes_of_removable:
                remove_from_saliences.append(tec_saliences[index])
                remove_from_tecs.append(tecs[index])

            for removable in remove_from_saliences:
                tec_saliences.remove(removable)

            for removable in remove_from_tecs:
                tecs.remove(removable)

    primary_tecs = []
    secondary_tecs = []

    for tec_list in selected_tecs:
        primary_tecs.append(tec_list[0])
        secondary_tecs.append(tec_list[1:len(tec_list)])

    return primary_tecs, secondary_tecs


def is_dataset_covered(P, d):
    for point in d:
        if point not in P:
            return False

    return True


def siatech_compress(d):
    """ Implements SIATECCompress as defined in [Meredith2016] but uses SIATECH. """

    tecs = siatech(d)
    sort_tecs_by_quality(tecs, d)

    covered_points = set()
    best_tecs = []
    for tec in tecs:
        new_points = tec.coverage() - covered_points
        if len(new_points) > len(tec.get_pattern()) + len(tec.get_translators()):
            best_tecs.append(tec)
            covered_points = covered_points | tec.coverage()
            if len(covered_points) == len(d):
                break

    residual = set(d) - set(covered_points)
    if residual:
        best_tecs.append(TEC(list(residual), [], []))

    return best_tecs


def cosiatech(d):
    """ Implements COSIATEC with SIATECH. """

    d = Dataset.sort_ascending(d)
    p = deepcopy(d)
    best_tecs = []

    while p:
        best_tec = get_best_tech(p, d)
        best_tecs.append(best_tec)
        p.remove_all(list(best_tec.coverage()))

    return best_tecs


def forths_algorithmh(d, c_min, sigma_min):
    """ Implements Forth's algorithm [Forth2012], [Meredith2016].
        Returns two lists of TECs, primary and secondary TECs. """

    tecs = siatech(d)
    tec_saliences = compute_tec_saliences(tecs, d)

    return forth_cover(d, tecs, tec_saliences, c_min, sigma_min)