"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os
import argparse
import numpy as np
from collections import defaultdict

from scipy.stats import ttest_ind


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dirs",
                        type=str,
                        help="The path of the experiments directories",
                        nargs='+',
                        required=True)

    arguments = parser.parse_args()
    return arguments


def load_experiment_results(experiment_directory):
    configurations_logs = os.listdir(experiment_directory)
    configurations_logs = [x for x in configurations_logs if x.startswith("log_augmented")]

    results = dict()
    for configurations_log in configurations_logs:
        with open(os.path.join(experiment_directory, configurations_log), 'r', encoding='utf8') as rp:
            performance_line = rp.readlines()[-2]
            performance_line_split = performance_line.split('\t')
            dev_performance = None
            test_performance = None
            try:
                dev_performance = float(performance_line_split[6])
                test_performance = float(performance_line_split[8])
            except:
                print(os.path.join(experiment_directory, configurations_log))

            if dev_performance and test_performance:
                results[configurations_log] = (dev_performance, test_performance)

    return results


def compute_average_per_configuration(experiments_results):
    configurations_list = experiments_results[list(experiments_results.keys())[0]].keys()
    seeds = experiments_results.keys()

    performances_per_configuration = defaultdict(list)
    for configuration in configurations_list:
        for seed in seeds:
            performance = experiments_results[seed][configuration]
            performances_per_configuration[configuration].append(performance)

    avg_performances = dict()
    for configuration in performances_per_configuration.keys():
        performances = performances_per_configuration[configuration]
        dev_performances = np.array([x[0] for x in performances])
        test_performances = np.array([x[1] for x in performances])

        avg_dev = np.mean(dev_performances)
        avg_test = np.mean(test_performances)
        std_dev = np.std(dev_performances)
        std_test = np.std(test_performances)

        avg_performances[configuration] = (avg_dev, std_dev, avg_test, std_test)

    return avg_performances


def get_bert_performance(experiment_directories):
    dev_performances = list()
    test_performances = list()
    for experiment_directory in experiment_directories:
        with open(os.path.join(experiment_directory, "log_original.txt"), 'r', encoding='utf8') as rp:
            log_lines = rp.readlines()
            performance_line = log_lines[-2]

            performance_line_split = performance_line.split('\t')
            if len(performance_line_split) < 7:
                continue
            dev_performance = float(performance_line_split[6])
            test_performance = float(performance_line_split[8])

            dev_performances.append(dev_performance)
            test_performances.append(test_performance)

    dev_performances = np.array(dev_performances)
    test_performances = np.array(test_performances)

    avg_dev = np.mean(dev_performances)
    std_dev = np.std(dev_performances)

    avg_test = np.mean(test_performances)
    std_test = np.std(test_performances)

    return avg_dev, std_dev, avg_test, std_test, dev_performances, test_performances


def main(args):
    experiment_directories = args.experiment_dirs

    bert_performances = get_bert_performance(experiment_directories)

    # dict will be seed -> configuration -> dev/test
    experiments_results = dict()
    for experiment_directory in experiment_directories:
        exp_results = load_experiment_results(experiment_directory)
        experiments_results[experiment_directory] = exp_results

    average_per_configuration = compute_average_per_configuration(experiments_results)
    sorted_performance_on_dev = sorted(average_per_configuration.items(), key=lambda x: -x[1][0])

    print("Model\tValidation Avg Acc\tValidation Std\tTest Avg Acc\tTest Std")
    print("\t".join(["BERT", str(bert_performances[0]), str(bert_performances[1]), str(bert_performances[2]), str(bert_performances[3])]))
    print("\t".join(["DATS"] + [str(x) for x in sorted_performance_on_dev[0][1]]))

    best_configuration_name = sorted_performance_on_dev[0][0]
    best_conf_metrics = list()
    for seed in experiments_results:
        metric = experiments_results[seed][best_configuration_name][1]
        best_conf_metrics.append(metric)

    test_result = ttest_ind(bert_performances[5], best_conf_metrics)
    print(test_result)


if __name__ == "__main__":
    args = parse_args()
    main(args)
