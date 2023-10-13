import os
import pickle
import itertools
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd

from evaluation.constants import DATASET_NAMES, INTERACTION_CATEGORIES, METRICS, METHOD_NAMES
from evaluation.utils import mkdir_if_missing


def load_pickle_files(args):
    results_dict = {}
    for metric, ec, dset, method in tqdm(list(
            itertools.product(args.metrics, args.evaluate_category, DATASET_NAMES, args.methods))):
        agg = metric.split('_')[0]
        metric = "_".join(metric.split('_')[1:])
        if ec:
            metric += "_cat"
        pickle_dir = os.path.join(args.eval_results_path, method, dset)
        try:
            pickle_file = os.path.join(pickle_dir, f'stats_{metric}_{agg}.pkl')
            with open(pickle_file, 'rb') as f:
                stats = pickle.load(f)
                avg = stats.avg
                if ec:
                    for cat_i, cat in enumerate(INTERACTION_CATEGORIES.values()):
                        key_tup = cat, method, dset, metric
                    if avg is None:
                        results_dict[key_tup] = np.nan
                    else:
                        results_dict[key_tup] = avg[cat_i]
                else:
                    key_tup = 'agg', method, dset, metric
                    if avg is None:
                        results_dict[key_tup] = np.nan
                    else:
                        results_dict[key_tup] =avg
        except FileNotFoundError:
            print(f"No file found for {dset} {method} {metric}-{agg} {'ec' if ec else ''} @ {pickle_file}")
            continue

    return results_dict


def main(args):
    # generate results tables
    results_dict = load_pickle_files(args)
    df = pd.DataFrame(list(results_dict.values()), index=pd.MultiIndex.from_tuples(list(results_dict.keys())))
    df = df.reset_index()
    df.columns = ['Category', 'Method', 'Dataset', 'Metric', 'Value']
    # get datasets as the columns, and add the ETC_UCY Avg. as a column
    df = df.pivot_table(index=['Category', 'Method', 'Metric'], columns='Dataset', values='Value')
    if len(df.index.get_level_values('Category').unique()) == 1:
        df.reset_index(level='Category', drop=True, inplace=True)
    mkdir_if_missing(args.save_results_path)
    for metric in df.index.get_level_values('Metric').unique():
        df.xs(metric, level='Metric').to_csv(os.path.join(args.save_results_path, f'{metric}.tsv'), sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', '-m', nargs='+', type=str, default=METHOD_NAMES)
    parser.add_argument('--datasets', '-d', nargs='+', type=str, default=DATASET_NAMES)
    parser.add_argument('--aggregations', '-a', nargs='+', type=str, default=['min', 'mean'])
    parser.add_argument('--evaluate_category', '-ec', nargs='+',
                        type=lambda x: True if x.lower()=="true" else False, default=[False])
    parser.add_argument('--metrics', '-mr', nargs='+', type=str, default=METRICS)
    parser.add_argument('--num_samples', '-ns', nargs='+', type=int, default=[20])
    parser.add_argument('--eval_results_path', default='results/evaluations')
    parser.add_argument('--save_results_path', default='results/results_tables')

    args = parser.parse_args()
    main(args)