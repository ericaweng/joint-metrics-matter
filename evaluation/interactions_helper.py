import numpy as np
import sys
from os import path
sys.path.append(path.dirname(path.abspath(__file__)))

#######################################
## Helper Functions for interactions ##
#######################################


def find_sublist(sublist, l):
    """finds index of start and end of sublist within l"""
    sublist_len = len(sublist)
    for ind in (i for i, e in enumerate(l) if e == sublist[0]):
        if np.all(l[ind:ind + sublist_len] == sublist):
            return ind, ind + sublist_len
    return None, None


def compute_speed(neigh_path, stride=1):
    """computes velocity magnitude of every ped in neigh_path
    neigh_path: (ts, *num_peds, 2)"""
    neigh_vel = neigh_path[stride:] - neigh_path[:-stride]  # velocity is the average over the past stride timesteps
    # have to make it the same length as position vector
    # neigh_vel = np.concatenate([np.zeros((stride, *neigh_path.shape[1:])), neigh_vel])
    pad_wid = [(stride, 0), (0, 0)]
    pad_wid = pad_wid + [(0, 0)] if len(neigh_path.shape) == 3 else pad_wid
    neigh_vel = np.pad(neigh_vel, pad_wid, mode='edge')
    # assert neigh_vel.shape == neigh_path.shape
    neigh_vel_mag = np.linalg.norm(neigh_vel, axis=-1)
    return neigh_vel_mag


def compute_speed_diff(path, neigh_path):
    """computes differences between the vel magnitudes of pp and neighbors"""
    prim_vel_mag = compute_speed(path)
    neigh_vel_mag = compute_speed(neigh_path)
    vel_diffs = np.abs(prim_vel_mag[:, np.newaxis] - neigh_vel_mag)
    return vel_diffs


def compute_velocity_interaction(path, neigh_path, stride=1):
    """Computes the angle between velocity of neighbours and velocity of pp"""
    prim_vel = path[
        stride:] - path[:
                        -stride]  # velocity is the average over the past 3 timesteps
    prim_vel = np.concatenate([np.zeros((stride, *path.shape[1:])), prim_vel])
    theta1 = np.arctan2(prim_vel[:, 1], prim_vel[:, 0])
    neigh_vel = neigh_path[stride:] - neigh_path[:-stride]
    neigh_vel = np.concatenate(
        [np.zeros((stride, *neigh_path.shape[1:])), neigh_vel])
    vel_interaction = np.zeros(neigh_vel.shape[0:2])
    sign_interaction = np.zeros(neigh_vel.shape[0:2])

    for n in range(neigh_vel.shape[1]):
        theta2 = np.arctan2(neigh_vel[:, n, 1], neigh_vel[:, n, 0])
        theta_diff = (theta2 - theta1) * 180 / np.pi
        theta_diff = theta_diff % 360
        theta_sign = theta_diff > 180
        sign_interaction[:, n] = theta_sign
        vel_interaction[:, n] = theta_diff
    return vel_interaction, sign_interaction


def compute_theta(path, stride=1):
    ## Computes the angle wrt to world coords of path
    prim_vel = path[
        stride:] - path[:
                        -stride]  # velocity is the average over the past 3 timesteps
    prim_vel = np.concatenate([np.zeros((stride, *path.shape[1:])), prim_vel])
    theta = np.arctan2(prim_vel[:, 1], prim_vel[:, 0])  # angle from horiz
    degrees = theta * 180 / np.pi  # angle difference at each ts
    return degrees


def compute_theta_interaction(path, neigh_path, stride=1):
    ## Computes the angle between line joining pp to neighbours and velocity of pp
    # path: (ts, np, 2)
    prim_vel = path[
        stride:] - path[:
                        -stride]  # velocity is the average over the past 3 timesteps
    prim_vel = np.concatenate([np.zeros((stride, *path.shape[1:])), prim_vel])
    theta1 = np.arctan2(prim_vel[:, 1], prim_vel[:, 0])  # angle from horiz
    rel_dist = neigh_path - path[:, np.
                                 newaxis, :]  # relative distance to each neighbor
    theta_interaction = np.zeros(rel_dist.shape[0:2])
    sign_interaction = np.zeros(rel_dist.shape[0:2])

    for n in range(rel_dist.shape[1]):  # for each ped
        theta2 = np.arctan2(
            rel_dist[:, n, 1],
            rel_dist[:, n, 0])  # angle from horiz for that non-main ped
        theta_diff = (theta2 -
                      theta1) * 180 / np.pi  # angle difference at each ts
        theta_diff = theta_diff % 360
        theta_sign = theta_diff > 180
        sign_interaction[:,
                         n] = theta_sign  # 1: going in the generally- towards each other direction else 0
        theta_interaction[:, n] = theta_diff
    return theta_interaction, sign_interaction


def compute_dist_rel(path, neigh_path):
    """Distance between primary ped (path) and neighbour (neigh_path)"""
    dist_rel = np.linalg.norm((neigh_path - path[:, np.newaxis, :]), axis=-1)
    return dist_rel


def is_angle_interaction_satisfied(theta_rel_orig, angle, angle_range):
    """determine if the interaction angle is in the correct range"""
    theta_rel = np.copy(theta_rel_orig)
    angle_low = angle - angle_range
    angle_high = angle + angle_range
    if angle_low < 0:
        theta_rel[np.where(
            theta_rel > 180)] = theta_rel[np.where(theta_rel > 180)] - 360
    if angle_high > 360:
        raise ValueError
    interaction_matrix = (angle_low < theta_rel) & (
        theta_rel <= angle_high) & (theta_rel < 500) == 1
    return interaction_matrix


def interaction_length_gt(interaction_matrix, length=1):
    interaction_sum = np.sum(interaction_matrix,
                             axis=0)  # sum across timesteps
    return interaction_sum >= length


def interaction_length_lt(interaction_matrix, length=1):
    interaction_sum = np.sum(interaction_matrix,
                             axis=0)  # sum across timesteps
    return interaction_sum < length


def check_int(path,
              neigh_path,
              dist_max=np.inf,
              pos_angle=0,
              vel_angle=0,
              pos_range=15,
              vel_range=15,
              speed_diff_max=np.inf,
              speed_diff_min=-np.inf,
              neigh_speed_min=-np.inf,
              neigh_speed_max=np.inf,
              self_speed_min=-np.inf,
              self_speed_max=np.inf,):
    """ main function that all interaction classifications use
    path: primary pedestrian (pp) 's trajectory (ts, 2)
    neigh_path: other pedestrians' trajectories (ts, num_ped, 2)
    dist_thresh: how close the agents have to be to be considered "interacting" (in m)
    pos_angle: the desired angle between pp's vel vector and the vector connecting pp to neighbor's pos
    vel_angle: the desired angle between pp's vel vector and neighbor's vel vector
    pos_range: how much the theta interaction angle can vary and still satisfy the requirement
               (accept interaction angle within pos_range of pos_angle)
               180 means you don't care about the pos_angle
    vel_range: how much the velocity interaction angle can vary and still satisfy the requirement
               (accept interaction angle within vel_range of vel_angle)
               180 means you don't care about the vel_angle
    vel_diff_threshold: how much the speeds between the pp and neighbor can differ
    vel_minimum: minimum speed for the pp
    vel_neigh_minimum: minimum speed for the neighbor
    vel_maximum: maximum speed for the pp
    vel_neigh_maximum: maximum speed for the neighbor
    Returns: (ts, num_peds) for each timestep, whether each ped satisfies the specified interaction with the primary
             pedestrian"""

    _, num_peds, _ = neigh_path.shape

    # angle interactions satisfied
    theta_interaction, _ = compute_theta_interaction(path, neigh_path)
    vel_interaction, _ = compute_velocity_interaction(path, neigh_path)
    pos_matrix = is_angle_interaction_satisfied(theta_interaction, pos_angle,
                                                pos_range)
    vel_matrix = is_angle_interaction_satisfied(vel_interaction, vel_angle,
                                                vel_range)

    # distance thresholds satisfied
    dist_matrix = compute_dist_rel(path, neigh_path) < dist_max

    # velocity difference thresholds satisfied
    speed_diff = compute_speed_diff(path, neigh_path)
    speed_diff_above_thresh = speed_diff_min < speed_diff
    speed_diff_below_thresh = speed_diff <= speed_diff_max
    speed_diff_matrix = speed_diff_above_thresh & speed_diff_below_thresh

    # neighbor velocity magnitude thresholds satisfied
    neigh_speed = compute_speed(neigh_path)
    neigh_speed_above_thresh = neigh_speed_min <= neigh_speed
    neigh_speed_below_thresh = neigh_speed <= neigh_speed_max
    neigh_speed_mat = neigh_speed_above_thresh & neigh_speed_below_thresh

    # self velocity magnitude thresholds satisfied
    self_speed = compute_speed(path)
    self_speed_above_thresh = self_speed_min < self_speed
    self_speed_below_thresh = self_speed <= self_speed_max
    self_speed_mat = self_speed_above_thresh & self_speed_below_thresh
    self_speed_mat = np.tile(self_speed_mat[:,None], (1, num_peds))
    assert self_speed_mat.shape == neigh_speed_mat.shape

    # combining it all
    interaction_matrix = pos_matrix & vel_matrix & dist_matrix & speed_diff_matrix & \
                         neigh_speed_mat & self_speed_mat

    return interaction_matrix
