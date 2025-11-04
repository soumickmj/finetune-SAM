#!/bin/bash

# Script to submit SLURM array jobs for fine-tuning SAM models
# This script generates job configurations and submits them as array jobs for better control

# ==============================================================================
# CONFIGURATION SECTION - Modify these variables as needed
# ==============================================================================

# Define the parameter values in arrays
# dsID_list=("254" "204" "259" "260")
# dsID_list=("204" "259" "260")
dsID_list=("259")

# load_all_masks=("True" "False") #whether to drop blank (for the selected classes) masks or not
load_all_masks=("False") #whether to drop blank (for the selected classes) masks or not

slice_indices=("2")

# splitTag_list=("Alex_final_Emma_final_trainN10_rs1701" "Alex_final_Emma_final_trainN20_rs1701")
splitTag_list=("Alex_final_Emma_final_trainN20_rs1701") #there are minor differences between 10 and 20, but 20 always outperformed 10, in some difficult cases with a big margin. 

# init_mode_list=("SAM" "MedSAM" "SSLSAM" "MedicoSAM" "MRIFoundation")
init_mode_list=("SAM" "MedSAM" "MedicoSAM") #SSLSAM and MRIFoundation did perform okay in some cases, but failed completely for others, and always performed worse compared to the other 3.

peft_mode_list=("adapter" "lora") #both are working comparatively. 

# Additional configuration
dsTag="EmmaAlexFinalV0"
# targets="liver,gallbladder,kidney,pancreas,aorta,spleen"
targets="liver,kidney,pancreas,aorta,spleen"
batch_size="3"
num_workers="3"
dsRoot="/group/glastonbury/soumick/dataset/ukbbnii/minisets"
out_type="ftSAM_AllexpGB"  # Name of the segmentation folder, as well as checkpoint folder 

# SLURM array job settings
max_concurrent_jobs=10  # Maximum number of jobs to run simultaneously
walltime="5:00:0"      # Job walltime
memory_per_cpu="2000Mb" # Memory per CPU
partition="gpuq"        # SLURM partition
mail_user="soumick.chatterjee@fht.org"

# ==============================================================================
# SCRIPT LOGIC - Generally no need to modify below this line
# ==============================================================================

# Generate timestamp for unique filenames
timestamp=$(date +"%Y%m%d_%H%M%S")
config_file="job_config_${timestamp}.txt"
log_dir="/group/glastonbury/soumick/SLURM/${out_type}"
mkdir -p "$log_dir"

echo "========================================================================"
echo "SLURM Array Job Submission Script for SAM Fine-tuning"
echo "========================================================================"
echo "Timestamp: $(date)"
echo "Configuration file: $config_file"
echo "Log directory: $log_dir"
echo ""

# Create the configuration file
echo "Generating job configuration..."
job_count=0

# Clear/create the config file
> "$config_file"

# Generate all parameter combinations
for dsID in "${dsID_list[@]}"; do
  for slice_index in "${slice_indices[@]}"; do
    for splitTag in "${splitTag_list[@]}"; do
      for init_mode in "${init_mode_list[@]}"; do
        for peft_mode in "${peft_mode_list[@]}"; do
          for load_all_mask in "${load_all_masks[@]}"; do
            
            # Determine expID based on splitTag
            if [[ "${splitTag}" == *"trainN10"* ]]; then
              expID="run2N10"
            elif [[ "${splitTag}" == *"trainN20"* ]]; then
              expID="run2N20"
            else
              expID="run2"
            fi

            # Write configuration to file
            # Format: expID|dsID|dsTag|splitTag|init_mode|peft_mode|targets|batch_size|num_workers|dsRoot|out_type|slice_index|load_all_mask
            echo "${expID}|${dsID}|${dsTag}|${splitTag}|${init_mode}|${peft_mode}|${targets}|${batch_size}|${num_workers}|${dsRoot}|${out_type}|${slice_index}|${load_all_mask}" >> "$config_file"
            
            job_count=$((job_count + 1))
            
          done
        done
      done
    done
  done
done

echo "Generated $job_count job configurations"
echo ""

# Display some example configurations
echo "Sample configurations (first 5):"
echo "=================================="
head -5 "$config_file" | nl
echo ""

if [ $job_count -gt 5 ]; then
    echo "... and $((job_count - 5)) more configurations"
    echo ""
fi

# Calculate array range
array_end=$((job_count - 1))

# Create job name with timestamp
job_name="${out_type}_array_${timestamp}"

echo "Job submission details:"
echo "======================="
echo "Job name: $job_name"
echo "Array range: 0-$array_end"
echo "Max concurrent jobs: $max_concurrent_jobs"
echo "Total jobs to run: $job_count"
echo "Walltime per job: $walltime"
echo "Memory per CPU: $memory_per_cpu"
echo "Partition: $partition"
echo ""

# Create the sbatch command
sbatch_cmd=(
    sbatch
    --job-name="$job_name"
    --array="0-${array_end}%${max_concurrent_jobs}"
    --time="$walltime"
    --nodes="1"
    --ntasks-per-node="1"
    --cpus-per-task="3"
    --gres="gpu:1"
    --mem-per-cpu="$memory_per_cpu"
    --chdir="$log_dir"
    --partition="$partition"
    --mail-type=ALL
    --mail-user="$mail_user"
    --export="CONFIG_FILE=$config_file"
    traintest_singlegpu_UKBAbd_array.sh
)

# Ask for confirmation before submitting
echo "About to submit array job with the following command:"
echo "${sbatch_cmd[*]}"
echo ""
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting array job..."
    
    # Submit the job
    job_output=$("${sbatch_cmd[@]}")
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Job submitted successfully!"
        echo "$job_output"
        
        # Extract job ID for monitoring
        job_id=$(echo "$job_output" | grep -o '[0-9]\+')
        
        echo ""
        echo "Monitoring commands:"
        echo "==================="
        echo "Check job status:    squeue -j $job_id"
        echo "Check array status:  squeue -j $job_id -t all"
        echo "Cancel all jobs:     scancel $job_id"
        echo "Cancel specific:     scancel ${job_id}_<task_id>"
        echo "Get the list of FAILED jobs: sacct -X -j $job_id -s FAILED --noheader --format=JobID"
        echo "View logs:           ls ${log_dir}/${out_type}_${job_name}_${job_id}_*.log"
        echo ""
        echo "Configuration file '$config_file' will be needed for the array job."
        echo "Do not delete it until all jobs complete!"
        
    else
        echo "✗ Failed to submit job (exit code: $exit_code)"
        echo "$job_output"
        exit 1
    fi
    
else
    echo "Job submission cancelled."
    echo "Configuration file '$config_file' has been created but job was not submitted."
    echo "You can review the configuration and run this script again."
fi

echo ""
echo "Script completed at $(date)"
echo "========================================================================"
