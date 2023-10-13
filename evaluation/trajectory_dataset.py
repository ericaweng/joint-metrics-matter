import math

import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self,
                 data_dir,
                 obs_len=8,
                 pred_len=12,
                 skip=1,
                 min_ped=1,):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len

        try:
            all_files = Path(self.data_dir).rglob("*.txt")

        except NotADirectoryError:
            all_files = [self.data_dir]

        # this will help demarcate where a sequence ends and a new one begins
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        sequence_names = []
        frame_ids = []
        for path in all_files:
            try:
                data = np.loadtxt(path, delimiter="\t")
            except ValueError:
                data = np.loadtxt(path, delimiter=" ")

            # Extract sequence names from file
            sequence_name = str(path).split('/')[-1].split('.')[0]

            frames = np.unique(data[:, 0]).tolist()
            # list of frames (time steps), each time step mapped all its pedestrians
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            # the number of unique trajectory sequences of length seq_len, overlap allowed
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx +
                                                          self.seq_len],
                                               axis=0)
                init_frame_id = curr_seq_data[:, 0][0]
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros(
                    (len(peds_in_curr_seq), 4, self.seq_len))
                # (num_peds, xy, traj_len) padded with zeros
                curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                # mask which shows which values are valid
                curr_loss_mask = np.zeros(
                    (len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                # trajectories (harder to predict)
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:,
                                                               1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # absolute index of the frame id of the first frame
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    # "" for the last frame
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    # skip this pedestrian if it doesn't exist in the frame for
                    # the entire duration of the sequence
                    if pad_end - pad_front != self.seq_len or np.unique(
                            curr_ped_seq[:, 0]).shape[0] != self.seq_len:
                        # sometimes, for any particular ped, there are frames
                        # missing in the middle
                        continue
                    # get only the x, y positions, then transpose so that x, y
                    # is the first axis
                    curr_ped_seq = np.transpose(curr_ped_seq)
                    # Make coordinates relative (delta between each timestep
                    # ~ action taken at each timestep)
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                # need at least min_ped pedestrians per scene to consider it
                if num_peds_considered >= min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    sequence_names.append(sequence_name)
                    frame_ids.append(init_frame_id)

        self.num_seq = len(seq_list)
        if len(seq_list) == 0:
            self.is_empty = True
            print(self.data_dir, "is empty or does not have any good trajectories")
            return

        self.is_empty = False
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.sequence_names = sequence_names
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def get_scenes(self):
        """returns scenes in shape (ts, num_peds, 2)"""
        scenes = [
            np.concatenate([obs, pred], axis=2).transpose(2, 0, 1)
            for (obs, pred, _, _, _, _, _) in self
        ]
        return scenes

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [  # (num_peds, 2, timesteps)
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            None, self.loss_mask[start:end, :],
            self.sequence_names[index]
        ]
        return out
