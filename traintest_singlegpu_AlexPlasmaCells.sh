#!/bin/bash
#SBATCH --job-name=ftSAM254V0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=gpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=1-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=3     # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --gres=gpu:1          # number of GPUs (max=4)
#SBATCH --chdir=/group/glastonbury/alex/yolov8_workspace/SAM_FT/SLURM
#SBATCH --output=ftSAM_%x_%j.log
#SBATCH --mem-per-cpu=4000Mb # RAM per CPU

cd $SLURM_SUBMIT_DIR

###process the arguments                                                                                                
#read the different keyworded commandline arguments
while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage
        exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"                                                                                                       shift                                                                                                               fi
        shift
    fi
    shift
done                                                 

#set the default values for the commandline arguments
expID="${expID:-init}"
dsID="${dsID:-Tiles}"
dsTag="${dsTag:-V0}"
dsRoot="${dsRoot:-/group/glastonbury/alex/yolov8_workspace/SAM_FT}"
splitTag="${splitTag:-Small}"
init_mode="${init_mode:-SAM}"
peft_mode="${peft_mode:-adapter}"
batch_size="${batch_size:-3}"
num_workers="${num_workers:-3}"

# Fixed params
arch="vit_b"

dataset_name="AlexPlasmaCells_${dsTag}"
train_img_list="${dsRoot}/CSVs/${dsID}/${splitTag}/train.csv"
val_img_list="${dsRoot}/CSVs/${dsID}/${splitTag}/val.csv"
test_img_list="${dsRoot}/CSVs/${dsID}/${splitTag}/test.csv"

# Dataset related parameters
if [ "$dsID" == "Tiles" ]; then    
    out_size="512"
elif [ "$dsID" == "Crops" ]; then
    out_size="96"
else
    echo "Unknown dataset ID: $dsID"
    exit 1
fi

num_cls=2

# Initial weights and normalisation
if [ "$init_mode" == "SAM" ]; then
    normalise_type="sam"
    sam_ckpt="pretrained_weights/SAM/sam_vit_b_01ec64.pth"
elif [ "$init_mode" == "MedSAM" ]; then
    normalise_type="medsam"
    sam_ckpt="pretrained_weights/MedSAM/medsam_vit_b.pth"
elif [ "$init_mode" == "SSLSAM" ]; then
    normalise_type="sam"
    sam_ckpt="pretrained_weights/SSLSAM/pretrain_encoderonly_mae_publicmri+breast_46.pth"
elif [ "$init_mode" == "PathoSAM" ]; then
    normalise_type="sam"
    sam_ckpt="pretrained_weights/PathoSAM/vit_b.pt"
else
    echo "Unknown initialization mode: $init_mode"
    exit 1
fi

# PEFT modes
if [ "$peft_mode" == "adapter" ]; then
    finetune_type="adapter"
    if_encoder_adapter="True"
    if_mask_decoder_adapter="True"
    if_encoder_lora_layer="False"
    if_decoder_lora_layer="False"
    lr="1e-4"
elif [ "$peft_mode" == "lora" ]; then
    finetune_type="lora"
    if_encoder_adapter="False"
    if_mask_decoder_adapter="False"
    if_encoder_lora_layer="True"
    if_decoder_lora_layer="True"
    lr="3e-4"
else
    echo "Unknown PEFT mode: $peft_mode"
    exit 1
fi

run_tag="${dsID}_${splitTag}_${expID}${init_mode}_${peft_mode}"

echo "----------------------------------------------------------------------"
echo "Variable values for the script:"
echo "----------------------------------------------------------------------"

echo "dataset_name:                     $dataset_name"
echo "test_img_list:                    $test_img_list"
echo "run_tag:                          $run_tag"
echo "arch:                             $arch"
echo "train_img_list:                   $train_img_list"
echo "val_img_list:                     $val_img_list"
echo "out_size:                         $out_size"
echo "num_cls:                          $num_cls"
echo "normalise_type:                   $normalise_type"
echo "sam_ckpt:                         $sam_ckpt"
echo "finetune_type:                    $finetune_type"
echo "if_encoder_adapter:               $if_encoder_adapter"
echo "if_mask_decoder_adapter:          $if_mask_decoder_adapter"
echo "if_encoder_lora_layer:            $if_encoder_lora_layer"
echo "if_decoder_lora_layer:            $if_decoder_lora_layer"
echo "lr:                               $lr"
echo "batch_size:                       $batch_size"
echo "num_workers:                      $num_workers"
echo "expID:                            $expID"
echo "dsID:                             $dsID"
echo "dsTag:                            $dsTag"
echo "dsRoot:                           $dsRoot"
echo "splitTag:                         $splitTag"  

echo "----------------------------------------------------------------------"
echo "End of variable check."
echo "----------------------------------------------------------------------"

# Setup the env
source /home/${USER}/.bashrc

# Fine-tune the model
# srun poetry run python SingleGPU_train_finetune_noprompt.py \
#     -if_warmup True \
#     -label_mapping "" \
#     -targets "" \
#     -arch "$arch" \
#     -dataset_name "$dataset_name" \
#     -train_img_list "$train_img_list" \
#     -val_img_list "$val_img_list" \
#     -out_size "$out_size" \
#     -num_cls "$num_cls" \
#     -normalize_type "$normalise_type" \
#     -sam_ckpt "$sam_ckpt" \
#     -finetune_type "$finetune_type" \
#     -if_encoder_adapter "$if_encoder_adapter" \
#     -if_mask_decoder_adapter "$if_mask_decoder_adapter" \
#     -if_encoder_lora_layer "$if_encoder_lora_layer" \
#     -if_decoder_lora_layer "$if_decoder_lora_layer" \
#     -lr "$lr" \
#     -b "$batch_size" \
#     -w "$num_workers" \
#     -run_tag "$run_tag" \
#     -dir_checkpoint "$dsRoot/checkpoints" 

# Test before finetuning
srun poetry run python val_finetune_noprompt.py \
    -test_prefinetune True \
    -dataset_name "$dataset_name" \
    -test_img_list "$test_img_list" \
    -run_tag "$run_tag" \
    -dir_checkpoint "$dsRoot/checkpoints"

# Validate the fine-tuned model
srun poetry run python val_finetune_noprompt.py \
    -dataset_name "$dataset_name" \
    -test_img_list "$test_img_list" \
    -run_tag "$run_tag" \
    -dir_checkpoint "$dsRoot/checkpoints" 