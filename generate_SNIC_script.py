# generate_SNIC_script.py
#!/usr/bin/env python3

from datetime import datetime

SNIC_SETUP = """#!/usr/bin/env bash
#SBATCH -A [REPLACE_WITH_YOUR_PROJECT_CODE] -p alvis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 1-00:00:00

echo "Hello cluster computing world!"

echo "JOB: ${SLURM_JOB_ID}"

echo "The following is RAM info."
free -h

echo "The following is GPU info."
nvidia-smi

echo "Launching experiments with singularity."

"""


def main():
    group_name = input('Please enter name for group of experiment (without any empty space).\n')
    # k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    k_list = [5, 7, 10, 20, 30]
    # run_ind_list = [i for i in range(50)]
    run_ind_list = [i for i in range(10)]
    # run_ind_list = [0]
    
    cmd_list = []
    for k in k_list:
        for run_ind in run_ind_list:
            cmd_str = f"singularity exec --nv graphbert.sif python3 batch_script_3_fine_tuning.py --dataset ogbn-arxiv --lr 0.01 --k {k} --max_epoch 200 --group_name {group_name} --run_index {run_ind}"
            cmd_list.append(cmd_str)

    now = datetime.now()
    date_time = now.strftime('%Y%m%d_%H%M')
    bash_script = f"./{group_name}_{date_time}_run_experiment.sh"
    print(f"Name of the bash script is {bash_script}")

    with open(bash_script, 'w') as f:
        f.write(SNIC_SETUP)
        for cmd in cmd_list:
            f.write(cmd + '\n')

if __name__ == "__main__":
    main()