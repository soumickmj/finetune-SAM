#!/bin/bash

# Define the parameter values in arrays
# dsID_list=("Crops" "Tiles")
dsID_list=("Tiles")

# splitTag_list=("Big" "Small")
splitTag_list=("Biggest")

# init_mode_list=("SAM" "MedSAM" "PathoSAM")
init_mode_list=("SAM")
# init_mode_list=("PathoSAM")

peft_mode_list=("adapter" "lora")
# peft_mode_list=("adapter")

# new loss params
loss_mode_list=(-1 0 1 2) 
add_boundary_loss_list=("True" "False") 
include_background_loss_list=("True" "False") 
val_dsc_monitor_list=("True" "False") 

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
              
                # --- Create a short, suitable job name ---
                if [ "${peft_mode}" == "adapter" ]; then peft_short="adp"; else peft_short="lora"; fi
                init_short="${init_mode,,}"

                # Shorten new parameters for job name
                loss_short="l${loss_mode}"
                if [ "${add_boundary_loss}" == "True" ]; then boundary_short="bndT"; else boundary_short="bndF"; fi
                if [ "${include_background_loss}" == "True" ]; then background_short="bgT"; else background_short="bgF"; fi
                if [ "${val_dsc_monitor}" == "True" ]; then val_short="vdscT"; else val_short="vdscF"; fi

                job_name="${peft_short}-${init_short}-${dsID}-${splitTag}-${loss_short}-${boundary_short}-${background_short}-${val_short}"

                expID="init" # This remains constant as per your original script

                # --- Construct and execute the command ---
                
                # 1. Build the command in an array for safe execution
                cmd=(
                    sbatch 
                    -J "V2${job_name}" 
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

echo "All jobs have been submitted."
