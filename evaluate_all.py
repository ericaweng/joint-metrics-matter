"""parallelize evaluate all"""
import os
import argparse
import subprocess
import torch
import multiprocessing
from pathlib import Path

from evaluation.constants import DATASET_NAMES, METHOD_NAMES


def get_cmds(args):
    cmd_i = 0
    if args.methods is None:
        methods = [m for m in METHOD_NAMES if 'gt' != m]
    else:
        methods = []
        for gs in args.methods:
            methods.extend([str(file).split('/')[-1] for file in Path('trajectories/').glob(f'*{gs}*')])
    if args.datasets is None:
        datasets = DATASET_NAMES
    else:
        datasets = args.datasets

    print("datasets:", datasets)

    metrics = " ".join(args.metrics)
    cmds_list = []
    for num_samples in args.num_samples:
        for dataset in datasets:
            for method in methods:
                cmd_i += 1
                if cmd_i < args.start_from:
                    continue
                ds = dataset
                tail = ""
                if method == 'ynet':
                    tail = " --ynet_ped --ynet_no_scene"
                cmd = f'python evaluation/evaluate.py{" -se" if args.skip_existing else ""} --method {method} --dataset {ds}{tail} ' \
                      f'--metrics {metrics} -ns {num_samples}'
                cmds_list.append(cmd)

    return cmds_list


def spawn(cmds, args):
    """launch cmds in separate threads, max_cmds_at_a_time at a time, until no more cmds to launch"""
    print(f"launching at most {args.max_cmds_at_a_time} cmds at a time:")

    sps = []
    num_gpus = len(args.gpus_available)
    total_cmds_launched = 0  # total cmds launched so far

    while total_cmds_launched < len(cmds):
        cmd = cmds[total_cmds_launched]
        # assign gpu and launch on separate thread
        gpu_i = args.gpus_available[total_cmds_launched % num_gpus]
        print(gpu_i, cmd)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_i)}
        if args.redirect_output:
            output_filename = 'logs_output'
            cmd = f"sudo {cmd} >> {output_filename}.txt 2>&1"
        # cmd = f"sudo {cmd}"
        sp = subprocess.Popen(cmd, env=env, shell=True)
        sps.append(sp)
        if len(sps) >= args.max_cmds_at_a_time:
            # this should work if all subprocesses take the same amount of time;
            # otherwise we might be waiting longer than necessary
            sps[0].wait()
            sps = sps[1:]
        total_cmds_launched += 1

    print("total cmds launched:", total_cmds_launched)
    [sp.wait() for sp in sps]
    print(f"finished all {total_cmds_launched} processes")


def main(args):
    cmds = get_cmds(args)
    spawn(cmds, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_cmds', '-mc', type=int, default=100)
    parser.add_argument('--max_cmds_at_a_time', '-c', type=int, default=max(1, multiprocessing.cpu_count()-3))
    parser.add_argument('--start_from', '-sf', type=int, default=0)
    try:
        cuda_visible = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    except KeyError:
        cuda_visible = list(range(torch.cuda.device_count()))
    parser.add_argument('--gpus_available', '-ga', nargs='+', type=int, default=cuda_visible)
    parser.add_argument('--redirect_output', '-ro', action='store_true')
    parser.add_argument('--methods', '-m', nargs='+', type=str, default=None)
    parser.add_argument('--datasets', '-d', nargs='+', type=str, default=None)
    parser.add_argument('--aggregations', '-a', nargs='+', type=str, default=['min', 'mean'])
    parser.add_argument('--metrics', '-mr', nargs='+', type=str, default=['ade', 'fde', 'joint_ade', 'joint_fde', 'col_pred', 'nll'])
    parser.add_argument('--num_samples', '-ns', nargs='+', type=int, default=[20])
    parser.add_argument('--skip_existing', '-se', action='store_true')
    parser.add_argument('--glob_str', nargs='+', default=None)
    parser.add_argument('--eval_results_path', default='results/evaluations')

    args = parser.parse_args()
    main(args)