"""
This file, when run as main, creates (w/ multiprocessing) and saves interaction masks for use during
interaction-category-based evaluation as txts of space-separated 0s and 1s as well as
prints interaction category statistics to tsv and screen
"""

import os
from collections import defaultdict
import argparse
import inspect
import configparser
from pathlib import Path
import multiprocessing

import numpy as np

from evaluation.constants import INTERACTION_CATEGORIES, INTERACTION_CAT_TO_FN, \
    DATASET_NAMES, DATASET_TYPES
from evaluation.trajectory_dataset import TrajectoryDataset
from evaluation.utils import mkdir_if_missing


def save_int_cats_to_tsv(tsv_file_path, print_table_list):
    with open(tsv_file_path, 'w') as file:
        line = "\t".join([
            INTERACTION_CATEGORIES[int_type]
            for int_type in INTERACTION_CATEGORIES
        ])
        line = f"N / proportion\t{line}\ttotal"
        all_to_print = line + "\n"
        all_dsets_total_trajs = 0
        for dset, dset_name in zip(print_table_list, DATASET_TYPES):
            print_string = []
            total_traj_sum = dset[0][1]
            for num_scenes_in_int, total_num_scenes in dset:
                prop = num_scenes_in_int / total_num_scenes if total_num_scenes > 0 else 0
                # each cell contains N and the proportion of all scenes that contain that int type
                line = f"{num_scenes_in_int} / {prop:.2f}"
                print_string.append(line)
                assert total_traj_sum == total_num_scenes or total_num_scenes == 0 or total_traj_sum == 0
                total_traj_sum = total_num_scenes
            print_string.append(
                f"{total_num_scenes:.0f}")  # total number of traj per dataset
            all_dsets_total_trajs += total_num_scenes
            line = "\t".join(print_string)
            line = f"{dset_name}\t{line}"  # single row of the table
            all_to_print += line + '\n'
        totals = np.sum(print_table_list, axis=0)
        totals_line = "\t".join([
            f"{int(tot):.0f} / {int(tot) / total_traj:.2f}"
            for tot, total_traj in totals
        ])
        totals_line = f"total\t{totals_line}\t{all_dsets_total_trajs:.0f}"
        all_to_print += totals_line
        print(all_to_print)
        file.write(all_to_print)

    print("done. output_dir:", tsv_file_path)


def get_hparam_dict(config_name):
    env_config = configparser.RawConfigParser()
    config_dir = 'datasets/interaction_configs'
    env_config.read(os.path.join(config_dir, f'{config_name}.cfg'))
    config = env_config.items(env_config.sections()[0])
    int_hparams_dict = {k: float(v.split(";")[0]) for k, v in config}
    return int_hparams_dict


def get_interaction_matrix_for_scene(scene, interaction_hparams, include_other):
    """scene: shape (ts=16/20, num_peds, 2)
    returns: int_cat_vec, shape (num_peds, num_int_cats) which maps each ped in the scene to its many-hot int_cat vector"""
    traj_len, num_peds, obs_dim = scene.shape
    assert obs_dim == 4

    datas = {int_cat: [] for int_cat in INTERACTION_CATEGORIES}
    datas['track_id'] = []
    datas['other'] = []
    infos = {int_cat: [] for int_cat in INTERACTION_CATEGORIES}
    infos['other'] = []

    for ped_i in range(num_peds):
        path = scene[:, ped_i, 2:4]
        track_id = scene[:, ped_i, 1][0]
        # All other agents
        neigh_path = np.concatenate(
            [scene[:, :ped_i, 2:4], scene[:, ped_i + 1:, 2:4]], axis=1)

        # iterate through the int_cats and add the interaction data and infos to a list
        for int_cat, int_cat_fn in INTERACTION_CAT_TO_FN.items():
            if int_cat == "other":
                continue
            hparams = {
                key: value
                for key, value in {
                    **interaction_hparams, 'neigh_path': neigh_path
                }.items() if key in inspect.getfullargspec(int_cat_fn).args
            }
            in_int_cat, info = int_cat_fn(path, **hparams)
            datas[int_cat].append(in_int_cat)
            assert isinstance(info, dict)
            infos[int_cat].append(info)
        datas['track_id'].append(track_id)
        datas['other'].append(~np.any([v for k, v in datas.items() if k in INTERACTION_CATEGORIES]))
        infos['other'].append({})

    int_cats_vec = []
    int_cats_vec.append(np.array(datas['track_id']).flatten())
    new_infos = {
        int_cat: defaultdict(list)
        for int_cat in INTERACTION_CATEGORIES
    }
    if include_other:
        int_cats_maybe_plus_other = INTERACTION_CATEGORIES + ['other']
    else:
        int_cats_maybe_plus_other = INTERACTION_CATEGORIES
    for int_cat in int_cats_maybe_plus_other:
        # for each ped, if they have at least one neighbor that causes them to fall into a certain category,
        # then classify that ped in this scene as falling into that category
        score = np.array(datas[int_cat]).flatten()
        int_cats_vec.append(score)# > 0)

        # move the int_cat axis to the front, peds axis to the back
        for info in infos[int_cat]:  # iterating over peds
            for key, value in info.items():  # iterating over int_cats
                new_infos[int_cat][key].append(value)

        # make an array
        for int_cat in new_infos:
            for key, value in new_infos[int_cat].items():
                new_infos[int_cat][key] = np.array(value)

    vec = np.array(int_cats_vec).T
    # print("vec =:", vec )
    return vec, new_infos


def apply(fn_args):
    interaction_hparams, data_item, data_i, include_other = fn_args
    (obs_traj, pred_traj_gt, _, _, _, _, sequence_name) = data_item
    obs_traj = obs_traj.numpy().transpose(2, 0, 1)
    pred_traj_gt = pred_traj_gt.numpy().transpose(2, 0, 1)
    scene = np.concatenate([obs_traj, pred_traj_gt], axis=0)
    frame_id = obs_traj[-1][0][0]

    int_cat_vec_scene, infos = get_interaction_matrix_for_scene(
            scene, interaction_hparams, include_other)

    save_id = int(frame_id * args.frame_scale)

    if args.save_masks:
        save_dir = f"{args.result_dir}/{sequence_name}"
        mkdir_if_missing(save_dir)
        np.savetxt(
                f"{save_dir}/frame_{save_id:06d}.txt",
                int_cat_vec_scene,
                fmt="%.1f")
    return int_cat_vec_scene[:, 1:]


def main(args):
    eth_ucy_hparams = get_hparam_dict('eth_ucy')
    sdd_hparams = get_hparam_dict('eth_ucy')
    interaction_hparams = [eth_ucy_hparams] * 5 + [sdd_hparams] * 4

    data_paths = list(Path(args.dataset_dir).glob("*"))
    print("data_paths:", data_paths)

    # table of the dataset N / proportions breakdown
    print_table_list_peds = np.zeros((len(data_paths), len(INTERACTION_CATEGORIES), 2))
    print_table_list_scenes = np.zeros((len(data_paths), len(INTERACTION_CATEGORIES), 2))
    num_peds_total = 0

    for dset_i, dset_path in enumerate(data_paths):
        dset_path = os.path.join(dset_path, 'test')
        dset = TrajectoryDataset(dset_path,
                                 obs_len=args.obs_len,
                                 pred_len=args.pred_len,
                                 skip=1,)

        if dset.is_empty:
            continue

        print("doing dataset:", dset_path, "\tdset len:", len(dset))

        list_of_arg_lists = []
        for data_i, data_item in enumerate(dset):
            args_list = interaction_hparams[dset_i], data_item, data_i, args.include_other
            list_of_arg_lists.append(args_list)

        num_peds_this_scene = np.sum([len(args_list[1][0]) for args_list in list_of_arg_lists])
        print("num scenes:", len(list_of_arg_lists))
        print("num peds:", num_peds_this_scene)
        num_peds_total += num_peds_this_scene
        if args.mp:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                int_cat_vecs = p.map(apply, list_of_arg_lists)
        else:
            int_cat_vecs = list(map(apply, list_of_arg_lists))

        print("finished getting int cats")

        # int_cat to list of scene_idxs that have at least one ped of that int_cat in the scene
        int_i_to_scene_in_int_cat = [
            [np.any(scene[:, int_i]) for scene in int_cat_vecs]
            for int_i in range(len(INTERACTION_CATEGORIES.keys()))
        ]  # (int_types, num_scenes)
        int_i_to_num_scenes = np.array([
            np.sum(scene_to_has_ped_in_int_cat)
            for scene_to_has_ped_in_int_cat in int_i_to_scene_in_int_cat
        ])  # (int_types,)

        int_i_to_num_peds = np.array([
            np.sum([np.sum(scene[:, int_i]) for scene in int_cat_vecs])
            for int_i in range(len(INTERACTION_CATEGORIES.keys()))
        ])  # (int_types,)

        # tables to save
        # N
        print_table_list_peds[dset_i, :, 1] = dset.loss_mask.shape[0]
        print_table_list_scenes[dset_i, :, 1] = len(dset)
        # proportions
        print_table_list_peds[dset_i, :, 0] = int_i_to_num_peds
        print_table_list_scenes[dset_i, :, 0] = int_i_to_num_scenes

        # for tracking which data_items belong to which int_cat
        for int_i, int_cat in enumerate(INTERACTION_CATEGORIES):
            print(f"int_i: {int_i}\tint_cat: {int_cat}\tn_peds: {int_i_to_num_peds[int_i]}\t"
                  f"n_scenes: {int_i_to_num_scenes[int_i]}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/raw_data')
    parser.add_argument('--dset_names', default=DATASET_NAMES, type=lambda x: x.split(","))
    parser.add_argument('--dset_types', type=str, nargs='+', default=DATASET_TYPES)
    # output / save options
    parser.add_argument('--dont_save_masks', dest='save_masks', default=True, action='store_false')
    parser.add_argument('--no_mp', dest='mp', default=True, action='store_false',
                        help="don't use multiprocessing")
    # Scenes
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--result_dir', type=str, default='results/interactions')
    # Stanford Drone Dataset
    parser.add_argument('--frame_scale', type=int, default=1.0)
    parser.add_argument('--include_other', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
