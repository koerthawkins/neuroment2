#!/bin/bash
#SBATCH --job-name="neuroment2"
#SBATCH --workdir=/clusterFS/home/student/koerthawkins/TIP/neuroment2/
#SBATCH --output=/clusterFS/home/student/koerthawkins/TIP/neuroment2/cluster/logs/slurm_%A-%a.log
# activate this line to redirect STDERR to separate log file
# SBATCH --error=/clusterFS/home/student/koerthawkins/TIP/neuroment2/cluster/errors/error_%j.err
# append output if file already exists
#SBATCH --open-mode=append
# number cpus per task
#SBATCH --cpus-per-task=1
# maximum time limit
# --time=DD-HH
# --time=HH:MM::SS
#SBATCH --time=07-00
# maximum memory limit
#SBATCH --mem=24G
# requires a gpu (otherwise: remove line)
# SBATCH --gres=gpu
# partitions to choose the nodes from (find e.g. using sview command)
# example for cpu only:
#SBATCH --partition=client,labor,short,simulation22
# example for gpu:
# SBATCH --partition=gpu,gpu2,gpu6
# if you are in the igpu partition (you will know when you are!)
# use the next three lines for running on ipgu. always exluce all hosts that are not yours or you have the okay of the owner to use that host!!
# SBATCH --partition=igpu
# SBATCH --exclude=sanderling,fritzfantom,ludwig,blackskimmer,calculus,anubis,ravenriad
# SBATCH --qos=igpu
# this can start multiple instances of the script with different array ids (env variable)
# one instance: --array=0
# three instances: --array=0-2
#SBATCH --array=0

export HOME="/clusterFS/home/student/koerthawkins"
export PYTHONPATH="/clusterFS/home/student/koerthawkins/TIP/neuroment2/"

# catch maximum error code of the python script
error=0; trap 'error=$(($?>$error?$?:$error))' ERR

# we simply forward all existing script args to train.py
/clusterFS/home/student/koerthawkins/miniconda3/envs/neuroment2-gpu/bin/python generate_data.py $@

exit $error

