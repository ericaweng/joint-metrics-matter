from evaluation.ped_interactions import *


RESULTS_DIR = f'./results/'

# ----------------------------- Datasets -----------------------------
DATASET_DISPLAY_NAMES = {
                      'eth': 'ETH',
                      'hotel': 'HOTEL',
                      'univ': 'UNIV',
                      'zara1': 'ZARA1',
                      'zara2': 'ZARA2',
                      'trajnet_sdd': 'TrajNet SDD'
                      }
DATASET_DISPLAY_NAMES_WITH_NUM_PEDS = {
        'eth': 'ETH \\footnotesize(1.4)',
        'hotel': 'HOTEL \\footnotesize(2.7)',
        'univ': 'UNIV \\footnotesize(25.7)',
        'zara1': 'ZARA1 \\footnotesize(3.3)',
        'zara2': 'ZARA2 \\footnotesize(5.9)',
        'trajnet_sdd': 'SDD TrajNet \\footnotesize(1.5)'
}
DATASET_GROUPS = [['eth', 'hotel', 'univ', 'zara1', 'zara2'], ['trajnet_sdd']]
DATASET_NAMES = list(DATASET_DISPLAY_NAMES.keys())
SCENE_NAMES_TO_NUM_FRAMES = {
        'biwi_eth': 253,
        'biwi_hotel': 445,
        'students001': 425,
        'students003': 522,  # univ 947 scenes total
        'crowds_zara01': 705,
        'crowds_zara02': 998, }
SEQUENCE_NAMES = {
    'eth': ['biwi_eth'],
    'hotel': ['biwi_hotel'],
    'zara1': ['crowds_zara01'],
    'zara2': ['crowds_zara02'],
    'univ': ['students001', 'students003'],
    'trajnet_sdd': [
            'coupa_0', 'coupa_1', 'gates_2', 'hyang_0', 'hyang_1', 'hyang_3',
            'hyang_8', 'little_0', 'little_1', 'little_2', 'little_3',
            'nexus_5', 'nexus_6', 'quad_0', 'quad_1', 'quad_2', 'quad_3',
    ],
}
N_DATASETS = len(DATASET_NAMES)
DATASET_TYPES = ['train', 'val', 'test']

# ----------------------------- Methods -------------------------------
METHOD_NAMES = ['sgan',
                'trajectron',
                'pecnet',
                'ynet',
                'memonet',
                'view_vertically',
                'joint_vv',
                'agentformer',
                'joint_agentformer',
                'gt'
]

# ----------------------------- Metrics -----------------------------
METRICS = ['min_joint_ade', 'min_joint_fde', 'mean_col_pred', 'max_col_pred', 'min_ade', 'min_fde']
AGGREGATIONS = ['min', 'mean', 'max']

# ----------------------------- Ped Interaction Categories -----------------------------
INTERACTIONS_PATH = './results/interactions'

INTERACTION_CATEGORIES = {
        's': 'static',
        'l': 'linear',
        'nl': 'non-linear',
        'lf': 'leader-follower',
        'grp': 'group',
        'ca': 'collision avoidance',
}

N_CATEGORIES = len(INTERACTION_CATEGORIES)
INTERACTION_CAT_FNS = [globals()[f'is_{name.replace(" ", "_").replace("-", "_")}'] for abbr, name in INTERACTION_CATEGORIES.items()]

INTERACTION_CAT_TO_FN = {
    a: b
    for a, b in zip(INTERACTION_CATEGORIES.keys(), INTERACTION_CAT_FNS)
}