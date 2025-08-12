import time
from datetime import datetime
from copy import deepcopy
from dataset import Dataset
from operator import itemgetter


def measure_function(measured_function, input_data, iterations, algorithm_name='', printout=True):
    """ Measures the average runtime of a function.
        measured_function is run on input_data iterations number of times to compute the average runtime. """
    times = []

    for _ in range(iterations):
        data = deepcopy(input_data)
        elapsed, result = measure_time(measured_function, data)
        times.append(elapsed)

    avg_time = 0
    for t in times:
        avg_time += t

    avg_time /= len(times)

    if algorithm_name == '':
        algorithm_name = measured_function.__name__

    if printout:
        print('Algorithm: ' + algorithm_name + '\nDataset: '
              + input_data.get_name() + ' (n=' + str(len(input_data)) + ')' + '\nAverage time: ' + str(avg_time)
              + '\nResult size: ' + str(len(result)) + '\n')

    return avg_time, result


def measure_time(measured_function, input_data):
    """ Measure the execution time of measured_function on input_data. """
    start = time.time()
    result = measured_function(input_data)
    end = time.time()
    elapsed = end - start
    return elapsed, result


def measure_function_time_on_datasets(measured_function, dataset_information, dataset_type, algorithm_name=''):
    """ Measure function runtime on multiple datasets and output the measurements into a csv file.
        dataset_information is a dictionary, where the keys are the input file paths and the values are the number
        of repetitions to run on that dataset.
        dataset_type is a string that is used as a part of the output filename. """

    if algorithm_name == '':
        algorithm_name = measured_function.__name__

    outputrows = []

    datasets = []
    for dataset_name in dataset_information:
        datasets.append((Dataset(dataset_name), dataset_information[dataset_name]))

    for dataset_pair in datasets:
        iterations = dataset_pair[1]
        dataset = dataset_pair[0]
        runtime, result = measure_function(measured_function, deepcopy(dataset), iterations, algorithm_name, printout=True)
        outputrows.append([algorithm_name, dataset.get_name(), str(len(dataset)), str(runtime), str(len(result))])

    outputrows.sort(key=itemgetter(3))

    timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())
    output_file_name = 'output/runtimes/' + algorithm_name + '-' + dataset_type + '-' + timestamp + '.csv'
    outputfile = open(output_file_name, mode='w')
    outputfile.write('algorithm, dataset, dataset_size, runtime, result_size\n')

    for outputrow in outputrows:
        outputfile.write(', '.join(outputrow) + '\n')
