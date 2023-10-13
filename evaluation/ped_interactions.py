""" Categorizes the Interaction """
import numpy as np

from evaluation.interactions_helper import (check_int, compute_dist_rel,
                                            compute_speed,
                                            interaction_length_gt,
                                            interaction_length_lt,
                                            find_sublist)


####################################
# Interaction categorization fns ##
####################################
# every interaction function implements the following structure:
def _function_name(path, **kwargs):
    """
    path: primary pedestrian (pp) 's trajectory (ts, 2)
    neigh_path: other pedestrians' trajectories (ts, num_ped, 2)
    kwargs: other keyword args needed by the interaction categorization fn
    returns: (is_this_cat, info) where is_this_cat is a bool which specifies if the ped given by path is part of this
             category, and info is a dict info relevant for plotting and visualization
    """
    pass


def is_static(path, static_dist_max, static_speed_max):
    """return 1 if distance between start and end position is within a certain threshold,
    and distance between each successive pair of positions is within a certain threshold"""
    start_end = np.linalg.norm(path[0] -
                               path[-1])  # distance between start and end pos
    mean_velocity = np.mean(compute_speed(path), axis=0)
    exists_interaction = start_end < static_dist_max and mean_velocity < static_speed_max
    return exists_interaction, {}


def is_moving_to_static(path, static_speed_max, moving_speed_min):
    """at least 2 timesteps of moving in a row occurs before 2 timesteps of static in a row"""
    static_matrix = compute_speed(path[8:]) < static_speed_max
    moving_matrix = compute_speed(path[8:]) > moving_speed_min
    _, end_of_static_section_i = find_sublist([True] * 2, moving_matrix)
    _, end_of_moving_section_i = find_sublist(
        [True] * 2, static_matrix[end_of_static_section_i:])
    exists_interaction = end_of_moving_section_i is not None and end_of_static_section_i is not None

    return exists_interaction, {}


def is_static_to_moving(path, static_speed_max, moving_speed_min):
    """at least 2 timesteps of static in a row occurs before 2 timesteps of moving in a row"""
    static_matrix = compute_speed(path[8:]) < static_speed_max
    moving_matrix = compute_speed(path[8:]) > moving_speed_min
    _, end_of_static_section_i = find_sublist([True] * 2, static_matrix)
    _, end_of_moving_section_i = find_sublist(
        [True] * 2, moving_matrix[end_of_static_section_i:])
    exists_interaction = end_of_moving_section_i is not None and end_of_static_section_i is not None

    return exists_interaction, {}


def is_static_and_moving(path, static_speed_max, moving_speed_min):
    """at least 2 timesteps of static in a row occurs, as well as 2 timesteps of moving in a row"""
    static_matrix = compute_speed(path[8:]) < static_speed_max
    moving_matrix = compute_speed(path[8:]) > moving_speed_min
    _, end_of_static_section_i = find_sublist([True] * 2, static_matrix)
    _, end_of_moving_section_i = find_sublist([True] * 2, moving_matrix)
    exists_interaction = end_of_moving_section_i is not None and end_of_static_section_i is not None
    return exists_interaction, {}


def is_speed_change(path, sc_speed_diff_min):
    speed = compute_speed(path[8:])
    speed_diff_over_thresh = np.max(speed) - np.min(speed) > sc_speed_diff_min
    return speed_diff_over_thresh, {}


def _non_linear(path, nl_resid_min, nl_degree=2):
    """
    Input:
    - traj: Numpy array of shape (traj_len, 2) or (traj_len, num_peds, 2)
    - threshold: Minimum sum of residuals (polyfit error) to be considered as a non-linear traj
    Output:
    - int: 1 -> Non-Linear 0-> Linear
    """
    t = np.arange(0, path.shape[0])
    px, res_x = np.polyfit(t, path[..., 0], nl_degree, full=True)[:2]
    py, res_y = np.polyfit(t, path[..., 1], nl_degree, full=True)[:2]
    not_linear = res_x + res_y
    # normalize by length of traj
    successive_pairs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    traj_length = np.sum(successive_pairs)
    not_linear = not_linear / (np.sqrt(traj_length) + 1e-12)
    not_linear = not_linear >= nl_resid_min
    poly = lambda a, b, c: lambda x: a * x**2 + b * x + c
    poly_fit_line = np.stack([poly(*px)(t), poly(*py)(t)], -1)
    return not_linear, {'poly_fit_line': poly_fit_line}


def is_linear(path, static_dist_max, static_speed_max, nl_resid_min,
              nl_degree):
    is_non_linear_vec, info = _non_linear(path, nl_resid_min, nl_degree)
    return (~is_static(path, static_dist_max, static_speed_max)[0]
            & ~is_non_linear_vec).item(), info


def is_non_linear(path, static_dist_max, static_speed_max, nl_resid_min,
                  nl_degree):
    is_non_linear_vec, info = _non_linear(path, nl_resid_min, nl_degree)
    return (~is_static(path, static_dist_max, static_speed_max)[0]
            & is_non_linear_vec).item(), info


def is_group(path, neigh_path, grp_mean_rel_dist_max, grp_std_rel_dist_max):
    """whether ego-agent is part of group with other agents,
    based on mean dist from agent over the course of the traj"""
    min_dist_satified = _is_group_dist(path, neigh_path, grp_mean_rel_dist_max,
                                       grp_std_rel_dist_max)
    positioning_satsified = _is_group_pos(path, neigh_path)
    is_group_bool = min_dist_satified & positioning_satsified
    return np.sum(is_group_bool.astype(np.int)) > 0, {
        'int_index': is_group_bool
    }


def _is_group_dist(path, neigh_path, grp_mean_rel_dist_max,
                   grp_std_rel_dist_max):
    """whether ego-agent is part of group with other agents,
    based on mean dist from agent over the course of the traj"""
    dist_rel = compute_dist_rel(path, neigh_path)
    mean_dist = np.mean(dist_rel, axis=0)
    std_dist = np.std(dist_rel, axis=0)
    mean_std_lt_thresh = (mean_dist < grp_mean_rel_dist_max) & (
        std_dist < grp_std_rel_dist_max)
    return mean_std_lt_thresh


def _is_group_pos(path, neigh_path):
    """whether ego-agent is part of group with other agents,
    based on relative positioning of the agents wrt to the ego-agent over the course of the traj"""
    ## Horizontal Position: walking side-by-side, neighbor is to either side of pp
    interaction_matrix_left = check_int(path,
                                        neigh_path,
                                        pos_angle=90,
                                        vel_angle=0,
                                        pos_range=45)  # (ts, np)
    interaction_matrix_right = check_int(path,
                                         neigh_path,
                                         pos_angle=270,
                                         vel_angle=0,
                                         pos_range=45)
    # todo right now it's only for at least 2 ts, should we check more?
    exist_side_neighs = (np.sum(interaction_matrix_left, axis=0)
                         | np.sum(interaction_matrix_right, axis=0)) > 1
    return exist_side_neighs


def _travelled_the_same_path(path,
                             neigh_path,
                             pp_is_leader=True,
                             min_ts_of_overlap=4,
                             path_match_thresh=0.2):
    """
    todo: implement with point-to-seg distance measuring
    returns array where each element is True if that neighbor has travelled the same path as the pp.
    Purpose is to capture non-linear leader-follower relationships.
    pp_is_leader: if True, searches for peds that have travelled the same path as pp, with pp being the leader
                  if False, searches for peds that have travelled the same path as pp, with pp being the follower
    """
    path = path.reshape(path.shape[0], 1, 2)
    assert path.shape[0] == neigh_path.shape[0]
    assert path.shape[2] == neigh_path.shape[2]
    path = np.tile(path, (1, neigh_path.shape[1], 1))
    assert path.shape == neigh_path.shape
    ts, num_peds, _ = neigh_path.shape
    interaction_index = np.full(neigh_path.shape[1], False)
    for diff_amount in range(1, ts - min_ts_of_overlap):
        if pp_is_leader:
            diff = path[:-diff_amount] - neigh_path[diff_amount:]
        else:
            diff = path[diff_amount:] - neigh_path[:-diff_amount]
        path_matches = np.linalg.norm(diff, axis=-1) < path_match_thresh
        interaction_index = interaction_index | interaction_length_gt(
            path_matches, length=min_ts_of_overlap)
    return interaction_index


def is_leader_follower(path, neigh_path, static_dist_max, static_speed_max, lf_dist_max):
    """ Identifying Leader Behavior """
    interaction_mat_obs = check_int( path[:8], neigh_path[:8], dist_max=lf_dist_max, pos_angle=180, )
    interaction_mat_pred = check_int( path[8:], neigh_path[8:], dist_max=lf_dist_max, pos_angle=180, )
    """ Identifying Follower Behavior """
    interaction_mat_obs_fol = check_int(path[:8], neigh_path[:8], dist_max=lf_dist_max)
    interaction_mat_pred_fol = check_int(path[8:], neigh_path[8:], dist_max=lf_dist_max)

    interaction_index_lead = (interaction_length_gt(interaction_mat_obs, length=4)
                        & interaction_length_gt(interaction_mat_pred, length=4)
                        | _travelled_the_same_path(path, neigh_path, pp_is_leader=False))

    interaction_index_fol = (interaction_length_gt(interaction_mat_obs_fol, length=4)
                        & interaction_length_gt(interaction_mat_pred_fol, length=4)
                        | _travelled_the_same_path(path, neigh_path))

    interaction_index = interaction_index_lead & interaction_index_fol & ~is_static(path, static_dist_max, static_speed_max)[0]
    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }


def is_leader(path, neigh_path, static_dist_max, static_speed_max, lf_dist_max,
              lf_speed_diff_max):
    """ Identifying Leader Behavior """
    interaction_mat_obs = check_int(
        path[:8],
        neigh_path[:8],
        dist_max=lf_dist_max,
        pos_angle=180,
    )
    interaction_mat_pred = check_int(
        path[8:],
        neigh_path[8:],
        dist_max=lf_dist_max,
        pos_angle=180,
    )
    interaction_index = (interaction_length_gt(interaction_mat_obs, length=4)
                        & interaction_length_gt(interaction_mat_pred, length=4)
                        | _travelled_the_same_path(path, neigh_path)) \
                        & ~is_static(path, static_dist_max, static_speed_max)[0]
    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }


def is_follower(path, neigh_path, static_dist_max, static_speed_max,
                lf_dist_max, lf_speed_diff_max):
    """ Identifying Follower Behavior """
    interaction_mat_obs = check_int(path[:8],
                                    neigh_path[:8],
                                    dist_max=lf_dist_max)
    interaction_mat_pred = check_int(path[8:],
                                     neigh_path[8:],
                                     dist_max=lf_dist_max)
    interaction_index = (interaction_length_gt(interaction_mat_obs, length=4)
                        & interaction_length_gt(interaction_mat_pred, length=4)
                        | _travelled_the_same_path(path, neigh_path, pp_is_leader=False)) \
                        & ~is_static(path, static_dist_max, static_speed_max)[0]

    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }


def is_passing(path, neigh_path, avoidance_dist_max, static_dist_max,
               static_speed_max):
    """ like avoidance, but both peds are moving and going the same direction """
    interaction_behind = check_int(path, neigh_path)
    interaction_in_front = check_int(path[8:], neigh_path[8:], pos_angle=180)
    interaction_left = check_int(
        path[8:],
        neigh_path[8:],
        pos_angle=90,
        vel_angle=0,  # pass to the left
        pos_range=45,
        dist_max=avoidance_dist_max)
    interaction_right = check_int(
        path[8:],
        neigh_path[8:],
        pos_angle=270,
        vel_angle=0,  # or to the right
        pos_range=45,
        dist_max=avoidance_dist_max)
    exist_side_neighs = np.any(interaction_left, axis=0) | np.any(
        interaction_right, axis=0)

    neigh_not_static = ~is_static(neigh_path[8:], static_dist_max,
                                  static_speed_max)[0]
    pp_not_static = ~is_static(path[8:], static_dist_max, static_speed_max)[0]

    interaction_index = interaction_length_gt(interaction_behind, length=2) \
                        & interaction_length_gt(interaction_in_front, length=2) \
                        & interaction_length_gt(exist_side_neighs, length=1) \
                        & interaction_length_lt(exist_side_neighs, length=5) \
                        & pp_not_static & neigh_not_static

    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }


def is_deference(path, neigh_path, static_dist_max, static_speed_max,
                 avoidance_dist_max, moving_speed_min):
    """opposite of single avoidance"""
    interaction_angles_met = check_int(path[8:],
                                       neigh_path[8:],
                                       pos_range=180,
                                       vel_range=180,
                                       dist_max=avoidance_dist_max,
                                       neigh_speed_min=moving_speed_min,
                                       self_speed_max=static_speed_max)

    neigh_not_static = ~is_static(neigh_path[8:], static_dist_max,
                                  static_speed_max)[0]

    interaction_index = interaction_length_gt(interaction_angles_met, length=1) \
                        & interaction_length_lt(interaction_angles_met, length=5) \
                        & neigh_not_static

    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }


def is_avoidance(path, neigh_path, static_dist_max, static_speed_max,
                 avoidance_dist_max, moving_speed_min):
    """ Identifying when the pp avoids a static ped"""
    interaction_angles_met = check_int(path[8:],
                                       neigh_path[8:],
                                       pos_range=180,
                                       vel_range=180,
                                       dist_max=avoidance_dist_max,
                                       neigh_speed_max=static_speed_max,
                                       self_speed_min=moving_speed_min)

    pp_not_static = ~is_static(path[8:], static_dist_max, static_speed_max)[0]

    interaction_index = interaction_length_gt(interaction_angles_met, length=1) \
                        & interaction_length_lt(interaction_angles_met, length=5) \
                        & pp_not_static

    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }

def is_collision_avoidance(path, neigh_path, static_dist_max, static_speed_max, avoidance_dist_max):
    """ Identifying Collision Avoidance Behavior """
    interaction_conds_met = check_int(path[8:],
                                      neigh_path[8:],
                                      pos_range=100,
                                      vel_angle=180,
                                      vel_range=135,
                                      # dist_max=avoidance_dist_max,
                                      )

    neigh_not_static = ~is_static(neigh_path[8:], static_dist_max,
                                  static_speed_max)[0]
    pp_not_static = ~is_static(path[8:], static_dist_max, static_speed_max)[0]
    interaction_index = interaction_length_gt(interaction_conds_met, length=1) \
                        & interaction_length_lt(interaction_conds_met, length=5) \
                        & neigh_not_static & pp_not_static

    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }


def is_mutual_avoidance(path, neigh_path, static_dist_max, static_speed_max,
                        avoidance_dist_max, moving_speed_min):
    """ Identifying Collision Avoidance Behavior """
    interaction_conds_met = check_int(path[8:],
                                      neigh_path[8:],
                                      pos_range=100,
                                      vel_angle=180,
                                      vel_range=135,
                                      dist_max=avoidance_dist_max,
                                      neigh_speed_min=moving_speed_min,
                                      self_speed_min=moving_speed_min)

    neigh_not_static = ~is_static(neigh_path[8:], static_dist_max,
                                  static_speed_max)[0]
    pp_not_static = ~is_static(path[8:], static_dist_max, static_speed_max)[0]
    interaction_index = interaction_length_gt(interaction_conds_met, length=1) \
                        & interaction_length_lt(interaction_conds_met, length=5) \
                        & neigh_not_static & pp_not_static

    return np.sum(interaction_index.astype(np.int)) > 0, {
        'int_index': interaction_index
    }


def is_other(is_non_linear, conds):
    vec = np.all([is_non_linear]) & np.all([~cond for cond in conds])
    return vec, {}


if __name__ == "__main__":
    pass
