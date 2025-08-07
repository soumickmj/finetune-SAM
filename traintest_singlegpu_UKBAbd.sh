#!/bin/bash
#SBATCH --job-name=ftSAM254V0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=soumick.chatterjee@fht.org
#SBATCH --partition=gpuq      # type of node we are using (cpuq or gpuq, this is not meant for interactive nodes)
#SBATCH --time=12:00:0      # walltime
#SBATCH --nodes=1             # number of nodes to be used
#SBATCH --ntasks-per-node=1   # number of tasks to run per node
#SBATCH --cpus-per-task=3     # number of CPUs per task (set it to greater than the number of workers, I would go for +1)
#SBATCH --gres=gpu:1          # number of GPUs (max=4)
#SBATCH --chdir=/group/glastonbury/soumick/SLURM/ftSAM
#SBATCH --output=ftSAM_%x_%j.log
#SBATCH --mem-per-cpu=2000Mb # RAM per CPU

cd $SLURM_SUBMIT_DIR

###process the arguments                                                                                                
#read the different keyworded commandline arguments
while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
        shift
    else
        shift
    fi
done                                                 

#set the default values for the commandline arguments
expID="${expID:-init}"
dsID="${dsID:-254}"
dsTag="${dsTag:-EmmaAlexFinalV0}"
dsRoot="${dsRoot:-/group/glastonbury/soumick/dataset/ukbbnii/minisets}"
splitTag="${splitTag:-Alex_final_Emma_final_trainN10_rs1701}"
targets="${targets:-liver}"
init_mode="${init_mode:-SAM}"
peft_mode="${peft_mode:-adapter}"
batch_size="${batch_size:-3}"
num_workers="${num_workers:-3}"

# Fixed params
arch="vit_b"

# Dataset related parameters
if [ "$dsID" == "254" ]; then
    dataset_name="UKB20254_${dsTag}"
    train_img_list="${dsRoot}/F20254_Liver_imaging_IDEAL_protocol_DICOM/manual_annotation/train_${splitTag}.csv"
    val_img_list="${dsRoot}/F20254_Liver_imaging_IDEAL_protocol_DICOM/manual_annotation/val_${splitTag}.csv"
    test_img_list="${dsRoot}/F20254_Liver_imaging_IDEAL_protocol_DICOM/manual_annotation/test_${splitTag}.csv"
    slice_index="3"
    prenorm_type="window"
    prenorm_window_min_percentile="1"
    prenorm_window_max_percentile="99"
    out_size="256"
elif [ "$dsID" == "204" ]; then
    dataset_name="UKB20204_${dsTag}"
    train_img_list="${dsRoot}/F20204_Liver_Imaging_T1_ShMoLLI_DICOM/manual_annotation/train_${splitTag}.csv"
    val_img_list="${dsRoot}/F20204_Liver_Imaging_T1_ShMoLLI_DICOM/manual_annotation/val_${splitTag}.csv"
    test_img_list="${dsRoot}/F20204_Liver_Imaging_T1_ShMoLLI_DICOM/manual_annotation/test_${splitTag}.csv"
    slice_index="2"
    prenorm_type="window"
    prenorm_window_min_percentile="1"
    prenorm_window_max_percentile="95"
    out_size="384"
elif [ "$dsID" == "259" ]; then
    dataset_name="UKB20259_${dsTag}"
    train_img_list="${dsRoot}/F20259_Pancreas_Images_ShMoLLI_DICOM/manual_annotation/train_${splitTag}.csv"
    val_img_list="${dsRoot}/F20259_Pancreas_Images_ShMoLLI_DICOM/manual_annotation/val_${splitTag}.csv"
    test_img_list="${dsRoot}/F20259_Pancreas_Images_ShMoLLI_DICOM/manual_annotation/test_${splitTag}.csv"
    slice_index="1"
    prenorm_type="window"
    prenorm_window_min_percentile="1"
    prenorm_window_max_percentile="95"
    out_size="384"
elif [ "$dsID" == "260" ]; then
    dataset_name="UKB20260_${dsTag}"
    train_img_list="${dsRoot}/F20260_Pancreas_Images_gradient_echo_DICOM/manual_annotation/train_${splitTag}.csv"
    val_img_list="${dsRoot}/F20260_Pancreas_Images_gradient_echo_DICOM/manual_annotation/val_${splitTag}.csv"
    test_img_list="${dsRoot}/F20260_Pancreas_Images_gradient_echo_DICOM/manual_annotation/test_${splitTag}.csv"
    slice_index="3"
    prenorm_type="window"
    prenorm_window_min_percentile="1"
    prenorm_window_max_percentile="95"
    out_size="160"
else
    echo "Unknown dataset ID: $dsID"
    exit 1
fi

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
elif [ "$init_mode" == "MedicoSAM" ]; then
    normalise_type="sam"
    sam_ckpt="pretrained_weights/MedicoSAM/vit_b_medicosam.pt"
elif [ "$init_mode" == "MRIFoundation" ]; then
    normalise_type="medsam"
    sam_ckpt="pretrained_weights/MRIFoundation/mri_foundation.pth"
elif [ "$init_mode" == "SSLSAM" ]; then
    normalise_type="sam"
    sam_ckpt="pretrained_weights/SSLSAM/pretrain_encoderonly_mae_publicmri+breast_46.pth"
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

run_tag="${expID}${init_mode}_${peft_mode}_slc${slice_index}_win${prenorm_window_min_percentile}to${prenorm_window_max_percentile}"

base_path=$(dirname "$test_img_list")
base_path="${base_path%/manual_annotation*}" 
seg_save_dir="${base_path}/segmentations/ftSAM"
seg_save_dir="${seg_save_dir}/${run_tag}/${targets}"

echo "----------------------------------------------------------------------"
echo "Variable values for the script:"
echo "----------------------------------------------------------------------"

echo "dataset_name:                     $dataset_name"
echo "test_img_list:                    $test_img_list"
echo "seg_save_dir:                     $seg_save_dir"
echo "run_tag:                          $run_tag"
echo "targets:                          $targets"
echo "arch:                             $arch"
echo "train_img_list:                   $train_img_list"
echo "val_img_list:                     $val_img_list"
echo "slice_index:                      $slice_index"
echo "prenorm_type:                     $prenorm_type"
echo "prenorm_window_min_percentile:    $prenorm_window_min_percentile"
echo "prenorm_window_max_percentile:    $prenorm_window_max_percentile"
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
srun poetry run python SingleGPU_train_finetune_noprompt.py \
    $(bool_to_arg "if_warmup" "True") \
    --targets "$targets" \
    --arch "$arch" \
    --dataset_name "$dataset_name" \
    --train_img_list "$train_img_list" \
    --val_img_list "$val_img_list" \
    --slice_index "$slice_index" \
    --prenorm_type "$prenorm_type" \
    --prenorm_window_min_percentile "$prenorm_window_min_percentile" \
    --prenorm_window_max_percentile "$prenorm_window_max_percentile" \
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
    --run_tag "$run_tag"

# Test before finetuning
srun poetry run python val_finetune_noprompt.py \
    $(bool_to_arg "test_prefinetune" "True") \
    --dataset_name "$dataset_name" \
    --test_img_list "$test_img_list" \
    --seg_save_dir "${seg_save_dir/${run_tag}/prefinetune_${run_tag}}" \
    --run_tag "$run_tag"

# Validate the fine-tuned model
srun poetry run python val_finetune_noprompt.py \
    --dataset_name "$dataset_name" \
    --test_img_list "$test_img_list" \
    --run_tag "$run_tag" \
    --seg_save_dir "$seg_save_dir" 