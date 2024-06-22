import os
import yaml
from datetime import datetime
import argparse

# The scratch directory is not limited in quota
LOG_DIR_BASE_PATH = "/vol/scratch/jonathany/slurm_logs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', default='config.yaml')
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as fp:
        config = yaml.safe_load(fp)
    slurm_config = config['slurm_params']
    sbatch_command = 'sbatch'
    logdir = f"{LOG_DIR_BASE_PATH}/{config['run_name']}_{datetime.now().strftime('%y%m%d-%H%M%S')}" # ckpts and midi will be saved here
    config['logdir'] = logdir
    slurm_config['output'] = os.path.join(logdir, slurm_config['output'])
    slurm_config['error'] = os.path.join(logdir, slurm_config['error'])
    os.makedirs(logdir, exist_ok=True)
    new_yaml_path = os.path.join(logdir, 'run_config.yaml')
    with open(new_yaml_path, 'w') as fp:
        yaml.dump(config, fp)
    
    for param in slurm_config:
        sbatch_command += f' --{param}={slurm_config[param]}'
    sbatch_command += f" {config['command']} --logdir {logdir} --yaml_path {new_yaml_path}"
    # print("sbatch command - ", sbatch_command)
    os.system(sbatch_command)
    