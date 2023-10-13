# Joint Metrics Matter: A Better Standard for Trajectory Forecasting Evaluation
This repository contains the code for the ICCV 2023 paper: 
[Joint Metrics Matter: A Better Standard for Trajectory Forecasting](https://arxiv.org/abs/2305.06292).

SOTA Trajectory Forecasting baselines like AgentFormer optimize for per-agent minimum displacement error metrics such as ADE. 
Our method, Joint AgentFormer is optimized for multi-agent minimum displacement error metrics such as JADE -- _Joint_ ADE.

![Joint AgentFormer](https://github.com/ericaweng/Joint_AgentFormer/assets/12485287/8c151916-82d7-45d6-9842-25c15f3c3d45)

We evaluate 6 baselines with respect to joint metrics (described in Sec. 3 of the paper) 
and perform improvements with respect to 2 (AgentFormer and View Vertically). 
The code for all evaluated methods (S-GAN, Trajectron++, View Vertically, MemoNet, 
AgentFormer, Joint AgentFormer) are added as submodules in the `methods/` directory.
Y-Net and PECNet are not yet included.

If you consider this work useful in your research, please cite our [paper](https://arxiv.org/abs/2305.06292):
```
@inproceedings{weng2023joint,
      title={Joint Metrics Matter: A Better Standard for Trajectory Forecasting}, 
      author={Weng, Erica and Hoshino, Hana and Ramanan, Deva and Kitani, Kris},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      year={2023}
}
```

## Overview
To evaluate methods with respect to our joint metrics, 
first we save the trajectories to text files in a standardized format. 
Then, we read from the files and evaluate them.

To evaluate your own trajectory forecasting method, add your method to `methods/` as a subdirectory
and write code to save all trajectories to `results/trajectories` in the standardized format
which is described in the next section.
Ensuring that your saved trajectory files follows the standardized format will 
ensure that you are evaluating the exact same dataset splits for fair comparison.

To create the training and evaluation datasets, we follow the leave-one-out training / evaluation 
setup that was used in the original [S-GAN](https://github.com/agrimgupta92/sgan).
That is, first we subdivide each dataset scene into sliding windows of length 20
(8 observation steps and 12 prediction steps) using stride of length 1 timestep.
We use all sequences with at least 1 pedestrian

## Trajectory Save Format
For each 20-timestep sequence, we save the observation (past) positions, 
ground-truth future positions, and predicted future positions to separate files. 
We save the observation to a file `obs.txt`, the ground-truth future to `gt.txt`, 
and the predicted futures to `sample_{sample_num}.txt`, 
where `{sample_num}` is the predicted sample number formatted with 3 digits, e.g. `{:03d}`. 
There are up to 20 predicted samples per sequence. 
Thus, there are up to 22 trajectory files saved per length-20 sequence.

We save these trajectory files to the path: `results/{method_name}/{dataset_environment_name}/{frame_id}/`
where `{frame_id}` is the frame number of the last observation step (t = 7) and serves as an identifying number for the length \delta t = 20 sequence. This format serves to ensure the dataset split being evaluated is the same across all methods.

each file is in the following format:
```
frame_id agent_id pos_x pos_y
...
```
In observation files there are 8 * `num_agents` lines; and in ground-truth and prediction
files there are 12 * `num_agents` lines. `num_agents` is the number of agents present in
that observation sequence. 

The positions are unnormalized and in world coordinates (the same frame as the ground
truth positions for ETH and our included version of Trajnet-SDD).

## Evaluation
After saving trajectories to the format described, please evaluate your method by 
running the following script from the repository root directory:

```
python evaluate.py --method <trajectory_save_relative_path> --dataset <dataset>
```

`trajectory_save_relative_path` is the relative path of the directory in which the
trajectories for the method are saved, inside `results/trajectories`;
e.g. `trajectron`, `sgan`, `view_vertically`, etc.

Evaluation results will be saved to `results/evaluations/<method_name>/<dataset>`as pickle files.

You can also run a script that will evaluate all methods in parallel: 

```
python evaluate_all.py
```

After all pickles are saved, then run: 

```
python evaluate_all.py
```

The final results table with all methods, datasets, metrics, and aggregations will be saved 
to `results/results_tables/<metric>.tsv`, one file for each metric.


# Datasets 
For ETH, we use the split used by S-GAN, which many trajectory forecasting works use. 
We include instructions for downloading the dataset for in the README for each method,
as each method may have different preprocessing steps.

For SDD, we use the Trajnet-SDD split released by the Trajnet Challenge.
The original test data is no longer available publicly, 
so we include the data in our repo in `datasets/raw_data/trajnet_sdd`. 
Note that the original data is in pixel-coordinates, but we processed the data according
to the estimated scales available in the [OpenTraj repo](https://github.com/crowdbotp/OpenTraj/blob/master/datasets/SDD/estimated_scales.yaml) 
to get the trajectories in approximate world coordinates.
However, in the paper, we report the pixel-coordinate results by inverse-transforming using the estimated scales.

# Pre-trained models
Pre-trained models are publicly available for S-GAN, View Vertically, AgentFormer, Joint AgentFormer,
and MemoNet via the submodules; see instructions for downloading them in the respective 
READMEs within the submodule for each method.

Joint View Vertically models will also be made available.
