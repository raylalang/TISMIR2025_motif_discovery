from dataset import Dataset
import orig_algorithms
import new_algorithms
import performance
from datetime import datetime
from operator import itemgetter
import heuristics


def measure_mtp_algorithms_on_datasets(datasets, datasets_name):
    """ Runs comparisons of SIA, SIAR, and SIAH.
        datasets is a dictionary, where the keys are the input file paths and the values are the number
        of repetitions to run on that dataset.
        datasets_name is used in naming the output files. """

    performance.measure_function_time_on_datasets(orig_algorithms.sia, datasets, datasets_name)
    r = 3
    performance.measure_function_time_on_datasets(lambda d: orig_algorithms.siar(d, r), datasets,
                                                  datasets_name, algorithm_name='siar(r=' + str(r) + ')')
    performance.measure_function_time_on_datasets(new_algorithms.siah, datasets, datasets_name)


def measure_tec_algorithms_on_datasets(datasets, datasets_name):
    """ Runs comparisons of SIATEC and SIATECH.
        datasets is a dictionary, where the keys are the input file paths and the values are the number
        of repetitions to run on that dataset.
        datasets_name is used in naming the output files. """

    performance.measure_function_time_on_datasets(orig_algorithms.siatec, datasets, datasets_name)
    performance.measure_function_time_on_datasets(new_algorithms.siatech, datasets, datasets_name)


def measure_siatech_siatechf(datasets):
    """ Runs comparisons of SIATECH, SIATECH-PF, and SIATECHF.
        datasets is a dictionary, where the keys are the input file paths and the values are the number
        of repetitions to run on that dataset. """

    performance.measure_function_time_on_datasets(new_algorithms.siatech, datasets, 'random_patterns')

    cr_th = 3
    performance.measure_function_time_on_datasets(lambda d: new_algorithms.siatechf(d, cr_th), datasets,
                                                  'random_patterns', algorithm_name='siatechf(th=' + str(cr_th) + ')')
    performance.measure_function_time_on_datasets(lambda d: new_algorithms.siatech_pf(d, cr_th), datasets,
                                                  'random_patterns', algorithm_name='siatech_pf(th=' + str(cr_th) + ')')

    cr_th = 5
    performance.measure_function_time_on_datasets(lambda d: new_algorithms.siatechf(d, cr_th), datasets,
                                                  'random_patterns', algorithm_name='siatechf(th=' + str(cr_th) + ')')


def measure_filtering_algorithms(datasets):
    """ Runs comparisons of SIATECHF and the point set compression algorithms.
        datasets is a dictionary, where the keys are the input file paths and the values are the number
        of repetitions to run on that dataset. """

    performance.measure_function_time_on_datasets(lambda d: orig_algorithms.forths_algorithmh(d, 15, 0.5), datasets,
                                                  'random_patterns', algorithm_name="forths_algo(15,0.5)")

    performance.measure_function_time_on_datasets(orig_algorithms.cosiatech, datasets, 'random_patterns')

    cr_th = 3
    performance.measure_function_time_on_datasets(lambda d: new_algorithms.siatechf(d, cr_th), datasets,
                                                  'random_patterns', algorithm_name='siatechf(th=' + str(cr_th) + ')')

    performance.measure_function_time_on_datasets(orig_algorithms.siatech_compress, datasets, 'random_patterns')


def compute_fraction_in_output(siatechf_output, other_output):
    """ Computes the precision of SIATECH by using other_output as the ground truth. """

    count = 0
    for tec in other_output:
        if tec in siatechf_output:
            count += 1

    return count / len(other_output)


def measure_filtering_algorithm_outputs(datasets):

    output_lines = []

    for dataset_name in datasets:
        dataset = Dataset(dataset_name)
        siatechf_output = new_algorithms.siatechf(dataset, 3)
        siatech_compr_output = orig_algorithms.siatech_compress(dataset)
        primary, secondary = orig_algorithms.forths_algorithmh(dataset, 15, 0.5)
        forths_output = primary
        for sec in secondary:
            if sec:
                forths_output += sec

        fraction_in_siatec_compr = compute_fraction_in_output(siatechf_output, siatech_compr_output)
        fraction_in_forths = compute_fraction_in_output(siatechf_output, forths_output)

        output_lines.append([dataset.get_name(), str(len(dataset)), 'siatec_compress', str(len(siatech_compr_output)),
                             str(len(siatechf_output)), str(fraction_in_siatec_compr)])
        output_lines.append([dataset.get_name(), str(len(dataset)), 'forths_algo', str(len(forths_output)),
                             str(len(siatechf_output)), str(fraction_in_forths)])

    # Write to file. Begin by putting the first line.
    timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())
    output_file_name = 'output/' + 'filtering_outputs_' + timestamp + '.csv'
    outputfile = open(output_file_name, mode='w')
    outputfile.write('dataset, dataset_size, other_algorithm, other_algo_output_size, siatechf_output_size, fraction_in_output\n')

    output_lines.sort(key=itemgetter(2, 1))

    for output_line in output_lines:
        outputfile.write(', '.join(output_line) + '\n')


def cr_statistics(tecs):
    """ Returns the minimum, maximum, and average compression ratio of TECs. """

    min_cr = 1000000
    max_cr = 0
    avg = 0

    for tec in tecs:
        cr = heuristics.compression_ratio(tec)
        if cr < min_cr:
            min_cr = cr
        if cr > max_cr:
            max_cr = cr

        avg += cr

    avg /= len(tecs)

    return min_cr, max_cr, avg


def compute_cr_statistics(datasets):
    """ Compute the compression ratios in the output of COSIATEC, SIATECCompress, Forth's algorithm,
        and SIATECHF on datasets. """

    output_lines = []

    for dataset_name in datasets:
        dataset = Dataset(dataset_name)
        siatech_compr_output = orig_algorithms.siatech_compress(dataset)
        primary, secondary = orig_algorithms.forths_algorithmh(dataset, 15, 0.5)
        # Put all TECs in primary and secondary into a single list.
        forths_output = primary
        for sec in secondary:
            if sec:
                forths_output += sec
        cosiatech_output = orig_algorithms.cosiatech(dataset)

        min_cr, max_cr, avg_cr = cr_statistics(siatech_compr_output)
        output_lines.append([dataset.get_name(), str(len(dataset)), 'siatec_compress', str(min_cr), str(max_cr), str(avg_cr)])

        min_cr, max_cr, avg_cr = cr_statistics(forths_output)
        output_lines.append([dataset.get_name(), str(len(dataset)), 'forths_algo', str(min_cr), str(max_cr), str(avg_cr)])

        min_cr, max_cr, avg_cr = cr_statistics(cosiatech_output)
        output_lines.append([dataset.get_name(), str(len(dataset)), 'cosiatec', str(min_cr), str(max_cr), str(avg_cr)])

    # Write to file. Begin by putting the first line.
    timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())
    output_file_name = 'output/' + 'filtering_outputs_' + timestamp + '.csv'
    outputfile = open(output_file_name, mode='w')
    outputfile.write('dataset, dataset_size, algorithm, min_cr, max_cr, avg_cr\n')

    output_lines.sort(key=itemgetter(2, 1))

    for output_line in output_lines:
        outputfile.write(', '.join(output_line) + '\n')
