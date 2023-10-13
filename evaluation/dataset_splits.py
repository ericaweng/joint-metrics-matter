
def get_ethucy_split(dataset, generated=False):
    seqs = [
        'biwi_eth', 'biwi_hotel', 'crowds_zara01', 'crowds_zara02',
        'crowds_zara03', 'students001', 'students003', 'uni_examples'
    ]

    if dataset in ['eth', 'eth_dyn']:
        test = ['biwi_eth']
    elif 'hotel' in dataset:
        test = ['biwi_hotel']
    elif 'zara1' in dataset:
        test = ['crowds_zara01']
    elif 'zara2' in dataset:
        test = ['crowds_zara02']
    elif 'univ' in dataset:
        test = ['students001', 'students003']
    elif dataset == 'gen':
        test = ['gen_test']
        seqs = ['gen']
    elif dataset == 'real_gen':
        test = ['biwi_eth', 'gen_test']
        seqs = [
            'biwi_eth', 'biwi_hotel', 'crowds_zara01', 'crowds_zara02',
            'crowds_zara03', 'students001', 'students003', 'uni_examples',
            'gen'
        ]
    elif dataset == 'eth_gen':
        test = ['eth_gen']

    if generated:
        test = [f"{d}_generated" for d in test]

    train, val = [], []
    for seq in seqs:
        if seq in test:
            continue
        train.append(f'{seq}_train')
        val.append(f'{seq}_val')
    return train, val, test


def get_stanford_drone_split():
    train = [
        'bookstore_0', 'bookstore_1', 'bookstore_2', 'bookstore_3', 'coupa_3',
        'deathCircle_0', 'deathCircle_1', 'deathCircle_2', 'deathCircle_3',
        'deathCircle_4', 'gates_0', 'gates_1', 'gates_3', 'gates_4', 'gates_5',
        'gates_6', 'gates_7', 'gates_8', 'hyang_4', 'hyang_5', 'hyang_6',
        'hyang_9', 'nexus_0', 'nexus_1', 'nexus_2', 'nexus_3', 'nexus_4',
        'nexus_7', 'nexus_8', 'nexus_9'
    ]
    val = [
        'coupa_2', 'hyang_10', 'hyang_11', 'hyang_12', 'hyang_13', 'hyang_14',
        'hyang_2', 'hyang_7', 'nexus_10', 'nexus_11'
    ]
    test = [
        'coupa_0', 'hyang_3', 'quad_3', 'little_2', 'nexus_5', 'quad_2',
        'gates_2', 'coupa_1', 'quad_1', 'hyang_1', 'hyang_8', 'little_1',
        'nexus_6', 'hyang_0', 'quad_0', 'little_0', 'little_3'
    ]

    return train, val, test
