#!/bin/bash
#SBATCH --job-name=ftSAM_array
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=gpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=5-00:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=3     # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --gres=gpu:1          # number of GPUs (max=4)
#SBATCH --chdir=/group/glastonbury/yolo_ibd_substructures/SLURM
#SBATCH --output=ftSAM_%x_%A_%a.log
#SBATCH --mem-per-cpu=2000Mb # RAM per CPU
#SBATCH --array=0-39%10      # Array job with max 10 concurrent jobs (will be set by submission script)

cd $SLURM_SUBMIT_DIR

# Read configuration file that maps array indices to parameters
config_file="${CONFIG_FILE:-job_config.txt}"

if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file $config_file not found!"
    exit 1
fi

# Read the line corresponding to this array task
line_number=$((SLURM_ARRAY_TASK_ID + 1))
config_line=$(sed -n "${line_number}p" "$config_file")

if [ -z "$config_line" ]; then
    echo "Error: No configuration found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse the configuration line (format: expID|dsID|dsTag|splitTag|init_mode|peft_mode|targets|batch_size|num_workers|dsRoot|out_type)
IFS='|' read -r expID dsID dsTag splitTag init_mode peft_mode targets batch_size num_workers dsRoot out_type <<< "$config_line"

# Set default values if not provided in config
dsTag="${dsTag:-EmaIBDTilesV0}"
dsRoot="${dsRoot:-/group/glastonbury/yolo_ibd_substructures}"

targets="${targets:-Lamina_propria,Crypt,Muscle,Surface,Granuloma,Granulation_tissue,Lymphoid_follicle_or_aggregates,Submucosa}"
batch_size="${batch_size:-3}"
num_workers="${num_workers:-3}"
out_type="${out_type:-ftSAM_all}"

# Fixed params
arch="vit_b"

echo "----------------------------------------------------------------------"
echo "SLURM Array Job - Task ID: $SLURM_ARRAY_TASK_ID"
echo "----------------------------------------------------------------------"
echo "Configuration: $config_line"
echo "----------------------------------------------------------------------"

# Dataset related parameters
dataset_name=${dsTag}
train_img_list="/group/glastonbury/yolo_ibd_substructures/train.csv"
val_img_list="/group/glastonbury/yolo_ibd_substructures/val.csv"
test_img_list="/group/glastonbury/yolo_ibd_substructures/test.csv"
label_mapping="/group/glastonbury/yolo_ibd_substructures/classlabel_mapping.pkl"
out_size="1024"

# Target parameters
#calculate num_cls based on targets by splitting the targets string with comma
IFS=',' read -r -a target_classes <<< "$targets"
num_cls=$((${#target_classes[@]} + 1))

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

run_tag="${dsID}_${expID}${init_mode}_${peft_mode}"

seg_save_dir="${dsRoot}/out_segmentations/${out_type}"
seg_save_dir="${seg_save_dir}/${run_tag}/${targets}"

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
echo "out_type:                         $out_type"  

echo "----------------------------------------------------------------------"
echo "End of variable check."
echo "----------------------------------------------------------------------"

# Setup the env
source /home/${USER}/.bashrc

# Function to convert boolean values to argument format for BooleanOptionalAction
bool_to_arg() {
    local var_name="$1"
    local var_value="$2"
    if [ "$var_value" == "True" ]; then
        echo "--$var_name"
    elif [ "$var_value" == "False" ]; then
        echo "--no-$var_name"
    else
        echo ""
    fi
}

# Fine-tune the model
srun /home/soumick.chatterjee/.local/bin/poetry run python SingleGPU_train_finetune_noprompt.py \
    --dir_checkpoint "$dsRoot/checkpoints" \
    $(bool_to_arg "if_warmup" "True") \
    --label_mapping "$label_mapping" \
    --targets "$targets" \
    --arch "$arch" \
    --dataset_name "$dataset_name" \
    --train_img_list "$train_img_list" \
    --val_img_list "$val_img_list" \
    --out_size "$out_size" \
    --num_cls "$num_cls" \
    --normalize_type "$normalise_type" \
    --sam_ckpt "$sam_ckpt" \
    --finetune_type "$finetune_type" \
    $(bool_to_arg "if_encoder_adapter" "$if_encoder_adapter") \
    $(bool_to_arg "if_mask_decoder_adapter" "$if_mask_decoder_adapter") \
    $(bool_to_arg "if_encoder_lora_layer" "$if_encoder_lora_layer") \
    $(bool_to_arg "if_decoder_lora_layer" "$if_decoder_lora_layer") \
    --lr "$lr" \
    --b "$batch_size" \
    --w "$num_workers" \
    --run_tag "$run_tag" \

# Test before finetuning
srun /home/soumick.chatterjee/.local/bin/poetry run python val_finetune_noprompt.py \
    --dir_checkpoint "$dsRoot/checkpoints" \
    $(bool_to_arg "test_prefinetune" "True") \
    --dataset_name "$dataset_name" \
    --test_img_list "$test_img_list" \
    --seg_save_dir "${seg_save_dir/${run_tag}/prefinetune_${run_tag}}" \
    --run_tag "$run_tag"

# Validate the fine-tuned model
srun /home/soumick.chatterjee/.local/bin/poetry run python val_finetune_noprompt.py \
    --dir_checkpoint "$dsRoot/checkpoints" \
    --dataset_name "$dataset_name" \
    --test_img_list "$test_img_list" \
    --run_tag "$run_tag" \
    --seg_save_dir "$seg_save_dir"
