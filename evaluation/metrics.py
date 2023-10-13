from functools import partial
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform, cdist
from itertools import starmap


def _lineseg_dist(a, b):
    """
    https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
    """
    # reduce computation
    if np.all(a == b):
        return np.linalg.norm(-a, axis=1)

    # normalized tangent vector
    d = np.zeros_like(a)
    # assert np.all(np.all(a == b, axis=-1) == np.isnan(ans))
    a_eq_b = np.all(a == b, axis=-1)
    d[~a_eq_b] = (b - a)[~a_eq_b] / np.linalg.norm(b[~a_eq_b] - a[~a_eq_b], axis=-1, keepdims=True)

    # signed parallel distance components
    s = (a * d).sum(axis=-1)
    t = (-b * d).sum(axis=-1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros_like(t)], axis=0)

    # perpendicular distance component
    c = np.cross(-a, d, axis=-1)

    ans = np.hypot(h, np.abs(c))

    # edge case where agent stays still
    ans[a_eq_b] = np.linalg.norm(-a, axis=1)[a_eq_b]

    return ans


def _get_diffs_pred(traj):
    """Same order of ped pairs as pdist.
    Input:
        - traj: (ts, n_ped, 2)"""
    num_peds = traj.shape[1]
    return np.concatenate([
        np.tile(traj[:, ped_i:ped_i + 1],
                (1, num_peds - ped_i - 1, 1)) - traj[:, ped_i + 1:]
        for ped_i in range(num_peds)
    ],
                          axis=1)


def _get_diffs_gt(traj, gt_traj):
    """same order of ped pairs as pdist"""
    num_peds = traj.shape[1]
    return np.stack([
        np.tile(traj[:, ped_i:ped_i + 1], (1, num_peds, 1)) - gt_traj
        for ped_i in range(num_peds)
    ],
                    axis=1)


def _check_collision_in_scene_fast(sample_idx,
                                   sample,
                                   gt_arr,
                                   ped_radius=0.15,
                                   scale=None):
    """sample: (num_peds, ts, 2) and same for gt_arr"""
    if scale is not None:
        # SDD reports in pixels. Thus, convert ped_radius to pixel.
        ped_radius /= scale

    sample = sample.transpose(1, 0, 2)
    gt_arr = gt_arr.transpose(1, 0, 2)
    ts, num_peds, _ = sample.shape
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    # pred
    # Get collision for timestep=0
    collision_0_pred = pdist(sample[0]) < ped_radius * 2
    # Get difference between each pair. (ts, n_ped_pairs, 2)
    ped_pair_diffs_pred = _get_diffs_pred(sample)
    pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)
    exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    collision_t_pred = _lineseg_dist(pxy, exy).reshape(
        ts - 1, num_ped_pairs) < ped_radius * 2
    collision_mat_pred = squareform(
        np.any(collision_t_pred, axis=0) | collision_0_pred)
    n_ped_with_col_pred_per_sample = np.any(collision_mat_pred, axis=0)
    # gt
    collision_0_gt = cdist(sample[0], gt_arr[0]) < ped_radius * 2
    np.fill_diagonal(collision_0_gt, False)
    ped_pair_diffs_gt = _get_diffs_gt(sample, gt_arr)
    pxy_gt = ped_pair_diffs_gt[:-1].reshape(-1, 2)
    exy_gt = ped_pair_diffs_gt[1:].reshape(-1, 2)
    collision_t_gt = _lineseg_dist(pxy_gt, exy_gt).reshape(
        ts - 1, num_peds, num_peds) < ped_radius * 2
    for ped_mat in collision_t_gt:
        np.fill_diagonal(ped_mat, False)
    collision_mat_gt = np.any(collision_t_gt, axis=0) | collision_0_gt
    n_ped_with_col_gt_per_sample = np.any(collision_mat_gt, axis=0)

    return sample_idx, n_ped_with_col_pred_per_sample, n_ped_with_col_gt_per_sample


def compute_ADE(pred,
                gt,
                mask,
                metrics_dict,
                aggregations=['min'],
                evaluate_category=False,
                **kwargs):
    """Compute average displacement error (ADE)
    Inputs:
        pred_arr: (list) A list of predicted trajectories.
                  Shape=(n_samples, future_timesteps, 4) where 4 is
                  [frame_id, agent_id, x, y].
        gt_arr: (list) A list of ground truth trajectories in respect to
                pred_arr. Shape=(future_timesteps, 4) where 4 is
                [frame_id, agent_id, x, y].
        aggregation: (str) The type of aggregation made at the end. Choices are
                     ['min', 'mean', 'max'].
        mask: (np.array) A bool array which indicates whether each pedestrian
              is in the interaction type. Shape=(n_agents, 15). The first value
              in each row is the agent_id.
    """
    metric = "ade_cat" if evaluate_category else "ade"
    diff = pred[:, :, 2:] - np.expand_dims(gt[:, 2:],
                                           axis=0)  # samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # samples x frames
    dist = dist.mean(axis=-1)  # samples

    for agg in aggregations:
        if agg == 'min':
            metrics_dict[metric][agg] += mask * dist.min(axis=0)  # (1, )
            # print(mask * dist.min(axis=0))
            # print(metrics_dict[metric][agg])
            # import ipdb; ipdb.set_trace()
        elif agg == 'mean':
            metrics_dict[metric][agg] += mask * dist.mean(axis=0)  # (1, )
        elif agg == 'max':
            metrics_dict[metric][agg] += mask * dist.max(axis=0)  # (1, )
        else:
            raise NotImplementedError()


def compute_FDE(pred,
                gt,
                mask,
                metrics_dict,
                aggregations=['min'],
                evaluate_category=False,
                **kwargs):
    """Compute final displacement error (FDE)
    Inputs:
        pred_arr: (list) A list of predicted trajectories.
                  Shape=(n_agents, n_samples, future_timesteps, 4) where 4 is
                  [frame_id, agent_id, x, y].
        gt_arr: (list) A list of ground truth trajectories in respect to
                pred_arr. Shape=(n_agents, future_timesteps, 4) where 4 is
                [frame_id, agent_id, x, y].
        aggregation: (str) The type of aggregation made at the end. Choices are
                     ['min', 'mean', 'max'].
        mask: (np.array) A bool array which indicates whether each pedestrian
              is in the interaction type. Shape=(n_agents, 15). The first value
              in each row is the agent_id.
        n_interactions: (int) Number of interaction types.
    """
    metric = "fde_cat" if evaluate_category else "fde"
    diff = pred[:, :, 2:] - np.expand_dims(gt[:, 2:],
                                           axis=0)  # samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # samples x frames
    dist = dist[..., -1]  # samples

    for agg in aggregations:
        if agg == 'min':
            metrics_dict[metric][agg] += mask * dist.min(axis=0)
        elif agg == 'mean':
            metrics_dict[metric][agg] += mask * dist.mean(axis=0)
        elif agg == 'max':
            metrics_dict[metric][agg] += mask * dist.max(axis=0)
        else:
            raise NotImplementedError()


def compute_ADE_joint(pred_arr,
                      gt_arr,
                      metrics_dict,
                      mask=1.,
                      aggregations=['max'],
                      return_samples=False,
                      evaluate_category=False,
                      **kwargs):
    """"""
    pred_arr = np.array(pred_arr)[..., 2:]
    gt_arr = np.array(gt_arr)[..., 2:]
    diff = pred_arr - np.expand_dims(gt_arr, axis=1)  # num_peds x samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # num_peds x samples x frames
    ade_per_ped_sample = dist.mean(axis=-1)  # num_peds x samples

    metric = "joint_ade_cat" if evaluate_category else "joint_ade"

    if evaluate_category:
        ade_per_ped_sample_cat = mask * ade_per_ped_sample.T[...,None]
        ade_per_sample_cat = ade_per_ped_sample_cat.sum(1)  # summed ade per sample, per cat
        # ade averaged across peds in cat, per sample
        num_peds_in_cat = metrics_dict[metric]['n']
        joint_ade_per_sample = ade_per_sample_cat / num_peds_in_cat  # (samples, n_cat)
    else:
        joint_ade_per_sample = ade_per_ped_sample.mean(axis=0)  # (samples, )

    for agg in aggregations:
        if agg == 'mean':
            metrics_dict[metric][agg] += joint_ade_per_sample.mean(axis=0)
        elif agg == 'min':
            metrics_dict[metric][agg] += joint_ade_per_sample.min(axis=0)
        elif agg == 'max':
            metrics_dict[metric][agg] += joint_ade_per_sample.max(axis=0)
    if return_samples:
        return joint_ade_per_sample


def compute_FDE_joint(pred_arr, gt_arr,
                      metrics_dict,
                      mask=1.,
                      aggregations=['max'],
                      evaluate_category=False,
                      **kwargs):
    pred_arr = np.array(pred_arr)[..., 2:]
    gt_arr = np.array(gt_arr)[..., 2:]
    diff = pred_arr - np.expand_dims(gt_arr, axis=1)  # num_peds x samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # num_peds x samples x frames
    fde_per_ped_sample = dist[...,-1]  # num_peds x samples

    metric = "joint_fde_cat" if evaluate_category else "joint_fde"

    if evaluate_category:
        fde_per_ped_sample_cat = mask * fde_per_ped_sample.T[...,None]
        fde_per_sample_cat = fde_per_ped_sample_cat.sum(1)  # summed fde per sample, per cat
        # fde averaged across peds in cat, per sample
        num_peds_in_cat = metrics_dict[metric]['n']
        joint_fde_per_sample = fde_per_sample_cat / num_peds_in_cat  # (samples, n_cat)
    else:
        joint_fde_per_sample = fde_per_ped_sample.mean(axis=0)  # samples

    for agg in aggregations:
        if agg == 'mean':
            metrics_dict[metric][agg] += joint_fde_per_sample.mean(axis=0)
        elif agg == 'min':
            metrics_dict[metric][agg] += joint_fde_per_sample.min(axis=0)
        elif agg == 'max':
            metrics_dict[metric][agg] += joint_fde_per_sample.max(axis=0)


def compute_CR(pred_arr,
               gt_arr,
               metrics_dict,
               mask=1.,
               ped_radius=0.1,
               aggregations=['max'],
               evaluate_category=False,
               **kwargs):
    """Compute collision rate and collision-free likelihood.
    Input:
        - pred_arr: (np.array) (n_pedestrian, n_samples, timesteps, 4)
        - gt_arr: (np.array) (n_pedestrian, timesteps, 4)
    Return:
        Collision rates
        col_pred: collision rate averaged across peds in a scene, then averaged across all scenes
        col_gt: collision rate averaged across peds in a scene, then averaged across all scenes
        cr: collision rate averaged across all peds in all scenes
    """
    # Collision rate between predicted trajectories.
    metric_pred = "col_pred_cat" if evaluate_category else "col_pred"

    pred_arr_ts_first = np.array(pred_arr).transpose(1, 0, 2, 3)[..., 2:]
    gt_arr_no_frames = np.array(gt_arr)[..., 2:]

    n_sample, n_ped, _, _ = pred_arr_ts_first.shape

    if evaluate_category:
        col_pred = np.zeros((n_sample, mask.shape[1]))
        col_gt = np.zeros((n_sample, mask.shape[1]))
    else:
        col_pred = np.zeros((n_sample))  # cr_pred
        col_gt = np.zeros((n_sample))  # cr_gt

    if n_ped > 1:
        r = starmap(partial(_check_collision_in_scene_fast, gt_arr=gt_arr_no_frames, ped_radius=ped_radius), enumerate(pred_arr_ts_first))
        for sample_idx, n_ped_with_col_pred, n_ped_with_col_gt in r:
            if evaluate_category:
                # For each category, compute collision rate
                col_pred_per_sample = mask * n_ped_with_col_pred.T[:, None]  # (n_peds, n_cat)
                col_gt_per_sample = mask * n_ped_with_col_gt.T[:, None]
                col_pred[sample_idx] = col_pred_per_sample.sum(axis=0) / metrics_dict[metric_pred]['n']  # (n_cat)
                col_gt[sample_idx] = col_gt_per_sample.sum(axis=0) / metrics_dict[metric_pred]['n']  # (n_cat)
            else:
                col_pred[sample_idx] = n_ped_with_col_pred.mean()
                col_gt[sample_idx] = n_ped_with_col_gt.mean()

    for agg in aggregations:
        if agg == 'mean':
            metrics_dict[metric_pred][agg] += col_pred.mean(axis=0)
        elif agg == 'min':
            samples = compute_ADE_joint(pred_arr, gt_arr, {}, aggregations=[], mask=1., return_samples=True, evaluate_category=False)
            best_SADE_idx = np.argmin(samples)
            metrics_dict[metric_pred][agg] += col_pred[best_SADE_idx]
        elif agg == 'max':
            metrics_dict[metric_pred][agg] += col_pred.max(axis=0)
        else:
            raise NotImplementedError()


def compute_NLL(pred,
                gt,
                metrics_dict,
                mask=1.,
                aggregations=['min'],
                evaluate_category=False,
                n_predictions=12,
                log_pdf_lower_bound=-20,
                **kwargs):
    """
     Inspired from https://github.com/StanfordASL/Trajectron.
    """
    metric = 'nll_cat' if evaluate_category else 'nll'

    n_sample, pred_len, _ = pred.shape
    pred = pred.transpose((0, 2, 1))  # (n_sample, state, timestep)
    # pred = pred.transpose((1, 0, 2))  # (pred_len, n_samples, 4)
    pred = pred[:, 2:4, :]
    # pred = pred[:, :, 2:4]
    gt = gt[:, 2:4]

    kde_ll = 0.
    for sample_idx in range(n_sample):
        for timestep in range(pred_len):
            try:
                kde = gaussian_kde(pred[sample_idx, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt[timestep].T),
                              a_min=log_pdf_lower_bound,
                              a_max=None)
                kde_ll += pdf / (pred_len * n_sample)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    for agg in aggregations:
        if agg == 'min':
            metrics_dict[metric][agg] = mask * kde_ll.mean()
        elif agg == 'mean':
            metrics_dict[metric][agg] += mask * kde_ll[0]
        elif agg == 'max':
            metrics_dict[metric][agg] += mask * 0
        else:
            raise NotImplementedError()
