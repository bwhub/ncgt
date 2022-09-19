# generate_benchmark_script.py
#!/usr/bin/env python3

import cmd
from datetime import datetime

def main():
    group_name = input('Please enter name for group of experiment (without any empty space).\n')
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    run_ind_list = [i for i in range(50)]
    # run_ind_list = [0]
    
    cmd_list = []
    for k in k_list:
        for run_ind in run_ind_list:
            # cmd_str = f"python3 batch_script_3_fine_tuning.py --dataset cora --lr 0.01 --k {k} --max_epoch 300 --group_name {group_name} --run_index {run_ind}"
            # cmd_str = f"python3 batch_script_3_fine_tuning.py --dataset citeseer --lr 0.001 --k {k} --max_epoch 2000 --group_name {group_name} --run_index {run_ind}"
            cmd_str = f"python3 batch_script_3_fine_tuning.py --dataset pubmed --lr 0.0005 --k {k} --max_epoch 500 --group_name {group_name} --run_index {run_ind}"
            cmd_list.append(cmd_str)

    now = datetime.now()
    date_time = now.strftime('%Y%m%d_%H%M')
    bash_script = f"./{group_name}_{date_time}_run_experiment.sh"
    print(f"Name of the bash script is {bash_script}")

    with open(bash_script, 'w') as f:
        f.write('#!/bin/bash\n\n')
        for cmd in cmd_list:
            f.write(cmd + '\n')

if __name__ == "__main__":
    main()