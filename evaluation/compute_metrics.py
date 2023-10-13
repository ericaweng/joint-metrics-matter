from collections import defaultdict
import numpy as np

from metrics import compute_ADE, compute_FDE, compute_CR, compute_NLL, \
    compute_ADE_joint, compute_FDE_joint
from constants import N_CATEGORIES

# Marginal Metrics (where each pedestrian is calculated individually)
SINGLE_METRIC_FN = {
    'ade': compute_ADE,
    'fde': compute_FDE,
    'nll': compute_NLL,
}

# Other metrics that cannot be defined in the above manner
OTHER_METRIC_FN = {
        'joint_ade': compute_ADE_joint,
        'joint_fde': compute_FDE_joint,
        'col_pred': compute_CR,
}

def create_mask(masks, pred_ids):
    new_masks = np.zeros((pred_ids.shape[0], 13))
    for idx, pred_id in enumerate(pred_ids):
        try:
            new_masks[idx] = masks[masks[:, 0] == pred_id][0, 1:]
        except IndexError:
            continue
    return new_masks


def compute_metrics(pred_arr,
                    gt_arr,
                    obs_arr=None,
                    aggregations=['min'],
                    masks=None,
                    ped_in_category=None,
                    evaluate_categories=False,
                    metrics=['ade', 'fde'],
                    ped_radius=0.3,
                    scale=None,
                    n_interactions=N_CATEGORIES):
    """Compute metrics in `metrics`
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
    metrics_dict = defaultdict(dict)
    n_samples = pred_arr[0].shape[0]
    for ec in evaluate_categories:
        for metric in metrics:
            # If evalute category, add "_cat" at the end of each metric
            metric_ = f"{metric}_cat" if ec else metric
            metrics_dict[metric_] = defaultdict(dict)
            # Calculate number of pedestrian. This will be used to take
            # the average later.
            if ec:
                ped_in_category = np.nan_to_num(ped_in_category, nan=0.0)
                metrics_dict[metric_]['n'] = ped_in_category
            else:
                metrics_dict[metric_]['n'] = len(pred_arr)
            for agg in aggregations:
                if ec:
                    if agg == 'all':
                        metrics_dict[metric_][agg] = np.zeros((n_interactions, n_samples))
                    else:
                        metrics_dict[metric_][agg] = np.zeros(n_interactions)
                else:
                    if agg == 'all':
                        metrics_dict[metric_][agg] = np.zeros(n_samples)
                    else:
                        metrics_dict[metric_][agg] = 0.0
                    mask = 1.

    pred_arr = np.stack(pred_arr)
    gt_arr = np.stack(gt_arr)
    if obs_arr is not None:
        obs_arr = np.stack(obs_arr)
    for p_i, (pred, gt) in enumerate(zip(pred_arr, gt_arr)):
        if obs_arr is not None:
            obs = obs_arr[p_i]
        else:
            obs = None
        if masks is not None:
            try:
                pred_id = np.unique(pred[:, :, 1])
                mask = masks[masks[:, 0] == pred_id][0, 1:]
            except IndexError:
                continue

        for ec in evaluate_categories:
            for metric in metrics:
                if metric == 'acl':
                    # This is calculated along speed so ignore
                    continue
                if metric not in SINGLE_METRIC_FN:
                    continue

                SINGLE_METRIC_FN[metric](pred=pred,
                                         gt=gt,
                                         obs=obs,
                                         mask=mask if ec else 1.,
                                         metrics_dict=metrics_dict,
                                         evaluate_category=ec,
                                         aggregations=aggregations)

    for ec in evaluate_categories:
        for metric in metrics:
            if metric not in OTHER_METRIC_FN:
                # Skip all other metrics
                continue

            ped_ids = [pred[0, 0, 1] for pred in pred_arr]

            masks_new = []
            for m in masks:
                if m[0] in ped_ids:
                    masks_new.append(m)
            if not masks_new:
                continue

            mask = np.stack(masks_new)

            pred_arr_new = []
            for pred in pred_arr:
                if pred[0, 0, 1] in mask[:, 0]:
                    pred_arr_new.append(pred)

            OTHER_METRIC_FN[metric](pred_arr=pred_arr_new,
                                    gt_arr=gt_arr,
                                    evaluate_category=ec,
                                    metrics_dict=metrics_dict,
                                    mask=mask[:, 1:] if ec else 1.,
                                    ped_radius=ped_radius,
                                    scale=scale,
                                    aggregations=aggregations)

    # Take average across n_pedestrian or n_samples
    # not needed, since all joint metrics first average across
    # peds in a scene, then across all scenes (that's what
    # makes them "joint")
    for ec in evaluate_categories:
        for metric in metrics:
            if metric in OTHER_METRIC_FN:
                continue
            metric = f"{metric}_cat" if ec else metric
            for agg in aggregations:
                if agg == 'all' and ec:
                    metrics_dict[metric][agg] /= metrics_dict[metric]['n'][:, None]
                else:
                    metrics_dict[metric][agg] /= metrics_dict[metric]['n']

    return metrics_dict
