#!/bin/bash

# Define the parameter values in arrays
# dsID_list=("Crops" "Tiles")
dsID_list=("Tiles")

# splitTag_list=("Big" "Small")
# splitTag_list=("Biggest")
splitTag_list=("Biggest_with_coords")

# init_mode_list=("SAM" "MedSAM" "PathoSAM")
init_mode_list=("SAM" "PathoSAM")
# init_mode_list=("PathoSAM")

peft_mode_list=("adapter" "lora")
# peft_mode_list=("adapter")

# new loss params
# loss_mode_list=(-1 0 1 2) 
loss_mode_list=(0 1 2) 
add_boundary_loss_list=("True" "False") 
include_background_loss_list=("True" "False") 
val_dsc_monitor_list=("False") 

# prompt_region_type_list=("random" "all")
prompt_region_type_list=("all")

#new additional params
# do_instance_training_list=("False" "True")
# no_bbox_input_list=("False" "True")
# prompt_dist_thre_ratio_list=("0.0" "0.1")

do_instance_training_list=("True")
no_bbox_input_list=("True" "False")
prompt_dist_thre_ratio_list=("0.0")

echo "Starting job submission script..."
echo "---"

# Loop through all combinations of the parameters
for dsID in "${dsID_list[@]}"; do
  for splitTag in "${splitTag_list[@]}"; do
    for init_mode in "${init_mode_list[@]}"; do
      for peft_mode in "${peft_mode_list[@]}"; do
        for loss_mode in "${loss_mode_list[@]}"; do
          for add_boundary_loss in "${add_boundary_loss_list[@]}"; do
            for include_background_loss in "${include_background_loss_list[@]}"; do
              for val_dsc_monitor in "${val_dsc_monitor_list[@]}"; do
                for prompt_region_type in "${prompt_region_type_list[@]}"; do
                  for do_instance_training in "${do_instance_training_list[@]}"; do
                    for no_bbox_input in "${no_bbox_input_list[@]}"; do
                      for prompt_dist_thre_ratio in "${prompt_dist_thre_ratio_list[@]}"; do

                        # --- Create a short, suitable job name ---
                        if [ "${peft_mode}" == "adapter" ]; then peft_short="adp"; else peft_short="lora"; fi
                        init_short="${init_mode,,}"

                        # Shorten new parameters for job name
                        loss_short="l${loss_mode}"
                        if [ "${add_boundary_loss}" == "True" ]; then boundary_short="bndT"; else boundary_short="bndF"; fi
                        if [ "${include_background_loss}" == "True" ]; then background_short="bgT"; else background_short="bgF"; fi
                        if [ "${val_dsc_monitor}" == "True" ]; then val_short="vdscT"; else val_short="vdscF"; fi
                        if [ "${do_instance_training}" == "True" ]; then instance_short="instT"; else instance_short="instF"; fi
                        if [ "${no_bbox_input}" == "True" ]; then bbox_short="nobbox"; else bbox_short="bbox"; fi

                        job_name="${peft_short}-${init_short}-${dsID}-${splitTag}-${loss_short}-${boundary_short}-${background_short}-${val_short}-${prompt_region_type}-${instance_short}-${bbox_short}-pdist${prompt_dist_thre_ratio}"

                        expID="init" # This remains constant as per your original script

                        # --- Construct and execute the command ---
                        
                        # 1. Build the command in an array for safe execution
                        cmd=(
                            sbatch 
                            -J "instRealBBox_${job_name}" 
                            traintest_singlegpu_AlexPlasmaCells.sh 
                            --expID "${expID}"
                            --dsID "${dsID}" 
                            --splitTag "${splitTag}" 
                            --init_mode "${init_mode}"
                            --peft_mode "${peft_mode}"
                            --loss_mode "${loss_mode}"
                            --add_boundary_loss "${add_boundary_loss}"
                            --include_background_loss "${include_background_loss}"
                            --val_dsc_monitor "${val_dsc_monitor}"
                            --use_bbox_training "True"
                            --prompt_region_type "${prompt_region_type}"
                            --do_instance_training "${do_instance_training}"
                            --no_bbox_input "${no_bbox_input}"
                            --prompt_dist_thre_ratio "${prompt_dist_thre_ratio}"
                        )

                        # 2. Print which job is being submitted for user feedback
                        echo "Submitting Job: ${job_name}"
                        echo "  Command -> ${cmd[*]}" # Show the full command being run

                        # 3. Execute the command to submit the job
                        "${cmd[@]}"
                        
                        echo "---"

                        # 4. Pause for a second to avoid overwhelming the scheduler
                        sleep 1
                        done
                      done
                    done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All jobs have been submitted."
