#!/bin/bash -l
#SBATCH --job-name=MammoCLIP   # Job name
#SBATCH --output=MammoCLIP/outputs.o%j # Name of stdout output file
#SBATCH --error=MammoCLIP/errors.e%j  # Name of stderr error file
#SBATCH --partition=small-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes 
#SBATCH --ntasks-per-node=1   # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --time=02:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001915  # Project for billing
cd <YOUR REPOSITORY HERE>
git pull || { echo "Git pull failed. Exiting."; exit 1; }
cd ~
## Load the needed GPU bindings
module use /appl/local/containers/ai-modules #switch as neeeded for your project
module load singularity-AI-bindings

# To have RCCL use the Slingshot interfaces:
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# To have RCCL use GPU RDMA:
export NCCL_NET_GDR_LEVEL=PHB
# Other stuff
export SIF=<YOUR DIRECTORY TO SIF HERE>
# Set variables
SEED=0


## Run your code
srun singularity exec \
    -B <YOUR REPOSITORY HERE> \
    -B <YOUR REPOSITORY HERE> \
    $SIF bash -c '$WITH_CONDA && export LD_LIBRARY_PATH=/opt/cray/pe/ccdb/5.0.3/lib:$LD_LIBRARY_PATH && source <YOUR REPOSITORY HERE> && which python && python <YOUR DIRECTORY HERE>/lumi_single_gpu_train_classifier.py \
      --data-dir <YOUR REPOSITORY HERE> \
      --img-dir <YOUR REPOSITORY HERE> \
      --csv-file <YOUR REPOSITORY HERE> \
      --data_frac 1.0 \
      --dataset "VinDr" \
      --arch "upmc_breast_clip_det_b5_period_n_ft" \
      --clip_chk_pt_path <YOUR REPOSITORY HERE> \
      --label "Suspicious_Calcification" \
      --epochs 30 \
      --batch-size 8 \
      --num-workers 0 \
      --print-freq 10000 \
      --log-freq 500 \
      --running-interactive "n" \
      --n_folds 1 \
      --lr 5.0e-5 \
      --weighted-BCE "y" \
      --balanced-dataloader "n" \
      --inference-mode "load" \
      --output_path <YOUR REPOSITORY HERE> \
      --tensorboard-path <YOUR REPOSITORY HERE> \
      --checkpoints <YOUR REPOSITORY HERE>'
