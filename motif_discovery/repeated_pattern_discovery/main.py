import measurements


def main():
    # Example of how to run measurements on the datasets.
    # Output files are written to the output directory.
    datasets = {'experiment_datasets/rand_patterns_in_filtering_tests/rand_patterns_500.csv': 1,
                'experiment_datasets/rand_patterns_in_filtering_tests/rand_patterns_1000.csv': 1}

    measurements.measure_siatech_siatechf(datasets)


if __name__ == "__main__":
    main()
