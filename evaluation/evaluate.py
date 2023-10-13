"""given predicted trajectories in results/trajectories, evaluates metrics
evaluates a single method and scene at a time
optionally evaluates by specified metric aggregation(s) {min, mean, max}
and interaction category {true, false}"""

import argparse
from collections import defaultdict
import os
import pickle
import yaml

import numpy as np
from tqdm import tqdm

from constants import METHOD_NAMES, DATASET_NAMES, METRICS
from compute_metrics import OTHER_METRIC_FN
from dataset_splits import get_ethucy_split, get_stanford_drone_split
from compute_metrics import compute_metrics
from utils import (AverageMeter, find_unique_common_from_lists, mkdir_if_missing, load_list_from_folder,)
import warnings
warnings.filterwarnings('ignore')



def do_one(data_file, seq_name, scaling, scale, evaluate_category, aggregations, metrics,):
    # Scene name should be in the form of `frame_xxxxxx`.
    scene_name = data_file.split('/')[-1]
    mask_path = os.path.join(args.interaction_masks_path, seq_name, f"{scene_name}.txt")
    try:
        mask = np.loadtxt(mask_path)
        if len(mask.shape) == 1:
            mask = np.expand_dims(mask, axis=0)
        ped_in_category = np.sum(mask[:, 1:], axis=0)
        ped_in_category = np.where(ped_in_category == 0, np.nan, ped_in_category)
    except OSError:
        return
        # raise OSError(f"mask_path: {mask_path}")

    trajectories_list, _ = load_list_from_folder(data_file)
    sample_all = []
    gt_all = []
    obs_all = []
    for traj_file in trajectories_list:
        if traj_file.endswith("latent_z.pt"):
            continue
        traj = np.loadtxt(traj_file, delimiter=' ', dtype='float32')
        if scaling:
            traj[:, 2:4] = traj[:, 2:4] / scale
        if 'gt' in traj_file:
            gt_all.append(traj)
        elif 'obs' in traj_file:
            obs_all.append(traj)
        elif 'sample' in traj_file and len(sample_all) < args.num_samples_to_eval:
            # (frames x agents, 4)
            sample_all.append(traj)
    # (n_samples, frames x agents, 4)
    if args.method == 'gt':
        sample_all = gt_all
    elif len(sample_all) == 0:
        raise ValueError(f"No samples found found for {data_file}")
    elif len(sample_all) < args.num_samples_to_eval:
        raise ValueError(f"not enough samples found for {data_file}")
    try:
        all_traj = np.stack(sample_all, axis=0)
    except ValueError:
        raise ValueError(f"path: {traj_file}\nshapes of samples for {data_file} do not match ({[a.shape for a in sample_all]})")
    # (agents x frames, 4)
    gt_all_traj = np.stack(gt_all, axis=0)
    if len(obs_all) > 0:
        obs_all_traj = np.stack(obs_all, axis=0)

    # Convert raw  data to our format for evaluation
    id_list = np.unique(all_traj[:, :, 1])
    agent_traj, gt_traj, obs_traj = [], [], []

    for idx in id_list:
        # GT traj. (frames, 4)
        gt_idx = gt_all_traj[gt_all_traj[:, :, 1] == idx]  # frames x 4
        if len(obs_all) > 0:
            obs_idx = obs_all_traj[obs_all_traj[:, :, 1] == idx]
            obs_traj.append(obs_idx)
        # predicted traj
        ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
        pred_idx = all_traj[:, ind, :]  # sample x frames x 4
        # filter data
        pred_idx, gt_idx = align_gt(pred_idx, gt_idx)
        agent_traj.append(pred_idx)
        gt_traj.append(gt_idx)

    if len(obs_traj) == 0:
        frame_skip = int(gt_traj[0][:, 0][1]) - int(gt_traj[0][:, 0][0])
        assert frame_skip != 0, f"frame_skip is 0. int(gt_traj[0][:, 0][1]) - int(gt_traj[0][:, 0][0]) = {int(gt_traj[0][:, 0][1]) - int(gt_traj[0][:, 0][0])}"
        frame_id = int(gt_traj[0][:, 0][0]) - frame_skip
        obs_traj = None
    else:
        frame_id = int(obs_traj[0][:, 0][-1])
    results = compute_metrics(agent_traj,
                              gt_traj,
                              obs_traj,
                              masks=mask,
                              ped_in_category=ped_in_category,
                              evaluate_categories=evaluate_category,
                              aggregations=aggregations,
                              metrics=metrics,
                              ped_radius=args.pedestrian_radius,
                              scale=scale)

    return results, frame_id, len(id_list)


def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(
        frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1]
    pred_new = pred[:, index_list2]
    return pred_new, gt_new

def main(args):
    args.evaluate_category = [
            True if ec.lower() == 'true' else False
            for ec in args.evaluate_category
    ]

    trajectories_dir = os.path.join(args.trajectories_dir, args.method)

    if 'sdd' in args.dataset:
        seq_train, seq_val, seq_test = get_stanford_drone_split()
        if args.dataset in os.listdir(trajectories_dir):
            trajectories_dir = os.path.join(trajectories_dir, args.dataset)
    elif args.dataset.startswith(
            ('eth', 'hotel', 'univ', 'zara1', 'zara2', 'gen', 'real_gen')):
        seq_train, seq_val, seq_test = get_ethucy_split(args.dataset)
    else:
        raise NotImplementedError()
    seq_eval = locals()[f"seq_{args.data_split}"]
    # Some methods combine univ sequence into one.

    # Metrics to evaluate
    metrics = args.metrics

    # Aggregation methods for each metric
    aggregations = args.aggregations

    # Whether to evaluate each interaction category or not.
    # When True, it will evaluate each interaction category.
    # Set False if not evaluating interaction categories
    evaluate_category = args.evaluate_category
    if 'true' in evaluate_category:
        print("make sure you have generated interaction category masks using python evaluation/generate_interaction_masks.py,"
              "o/w the program will not produce results for interaction category")

    stats_meter = defaultdict(dict)
    for ec in evaluate_category:
        for metric in metrics:
            if ec:
                metric = f"{metric}_cat"
            stats_meter[metric] = {}
            for agg in aggregations:
                stats_meter[metric][agg] = AverageMeter(
                        masked=ec, all_samples=True if agg == 'all' else False)

    output_dir = args.output_dir#f"{args.output_dir}_rad-{str(args.pedestrian_radius)}_samples-{args.num_samples_to_eval}"
    log_dir = os.path.join(output_dir, args.method, args.dataset)
    mkdir_if_missing(log_dir)

    # Scale world coordinates back to image coordinates
    scaling = False
    if (args.method in ['trajectron', 'sgan'] or 'af_' in args.method) and 'trajnet' in args.dataset:
        print("Scaling world coordinates back to image coordinates")
        scaling = True
        with open(args.scales_path, 'r') as hf:
            scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)

    for seq_name in seq_eval:
        # skip evaluation of already calucated results
        dataset = args.dataset
        pickle_dir = os.path.join(output_dir, args.method, dataset)
        if args.skip_existing and np.all([os.path.exists(os.path.join(pickle_dir, f'stats_{metric}_{agg}.pkl'))
                                          for metric in metrics for agg in aggregations]):
            print(f"Skipping {seq_name} for {args.method}")
            continue

        data_file_lists, _ = load_list_from_folder(os.path.join(trajectories_dir, seq_name))
        print('loading results from: ', os.path.join(trajectories_dir, seq_name))

        if scaling:
            # Scaling is only applied for TrajNet SDD
            scale = scales_yaml_content[seq_name.split('_')[0]][f"video{seq_name.split('_')[1]}"]['scale']
        else:
            scale = None

        # Each examples e.g., frame_0001 - frame_0020
        # per ped num stats
        for idx, data_file in enumerate(tqdm(data_file_lists, desc=seq_name, disable=args.no_tqdm)):
            results, frame_id, num_peds = do_one(data_file, seq_name, scaling, scale, evaluate_category, aggregations, metrics)
            for ec in evaluate_category:
                for metric in metrics:
                    if ec:
                        metric = f"{metric}_cat"
                    for agg in aggregations:
                        if metric in OTHER_METRIC_FN:
                            n = 1
                        else:
                            n = results[metric]['n']
                            if not ec:
                                assert num_peds == n
                        stats_meter[metric][agg].update(results[metric][agg],
                                                        mean_value=True,
                                                        n=n,
                                                        current_frame=frame_id)

    dataset = args.dataset

    pickle_dir = os.path.join(output_dir, args.method, dataset)
    mkdir_if_missing(pickle_dir)

    for metric in metrics:
        if ec:
            metric = f"{metric}_cat"
        for agg in aggregations:
            fn = os.path.join(pickle_dir, f'stats_{metric}_{agg}.pkl')
            if np.all(stats_meter[metric][agg].count) == 0:
                continue
            with open(fn, 'wb') as f:
                pickle.dump(stats_meter[metric][agg], f )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,  choices=METHOD_NAMES)
    parser.add_argument('--dataset', type=str, default='eth', choices=DATASET_NAMES)
    parser.add_argument('--metrics', nargs='+', type=str, default=METRICS)
    parser.add_argument('--aggregations', nargs='+', type=str, default=['min', 'mean', 'max'])
    parser.add_argument('--evaluate_category', nargs='+', type=str, default=['false'])
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--trajectories_dir', type=str, default='./results/trajectories')
    parser.add_argument('--num_samples_to_eval', '-ns', type=int, default=20)
    parser.add_argument('--interaction_masks_path', type=str, default='./results/interactions')
    parser.add_argument('--output_dir', type=str, default='./results/evaluations')
    # Stanford Drone Dataset
    parser.add_argument('--scales_path', type=str, default='./datasets/trajnet_sdd/estimated_scales.yaml')
    # Other
    parser.add_argument('--pedestrian_radius', type=float, default=0.1)
    parser.add_argument('--no_tqdm', action='store_true', help='Disable TQDM.')
    parser.add_argument('--skip_existing', '-se', action='store_true')
    args = parser.parse_args()

    main(args)
