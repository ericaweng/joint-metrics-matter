"""parallelize evaluate all"""
import subprocess


for dataset in ['eth', 'univ', 'hotel', 'zara1', 'zara2', 'trajnet_sdd']:
    for method in ['agentformer', 'pecnet', 'sgan', 'trajectron', 'ynet', 'vv', 'memonet', 'agentformer_sfm']:
        if method == 'trajectron':
            ds = f"{dataset}_dyn"
        else:
            ds = dataset
        tail = ""
        if method == 'ynet':
            tail = " --ynet_ped --ynet_no_scene"
        if dataset == 'zara1':  # 'univ' or dataset == 'eth':
            univ = '-1'
        else:
            univ = ''
        cmd = f'python experiments/save_predicted_trajectories.py --sequence_name {dataset} --cfg {dataset}_agentformer_sfm_pre8-2{univ} --method agentformer_sfm'
        print("cmd:", cmd)
        subprocess.Popen(cmd.split(' '))
