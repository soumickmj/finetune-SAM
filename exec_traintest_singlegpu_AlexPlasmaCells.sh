#!/bin/bash

# Define the parameter values in arrays
# dsID_list=("Crops" "Tiles")
dsID_list=("Crops")

splitTag_list=("Big" "Small")
# splitTag_list=("Biggest")

# init_mode_list=("SAM" "MedSAM" "PathoSAM")
init_mode_list=("SAM" "PathoSAM")
# init_mode_list=("PathoSAM")

peft_mode_list=("adapter" "lora")

echo "Starting job submission script..."
echo "---"

# Loop through all combinations of the parameters
for dsID in "${dsID_list[@]}"; do
  for splitTag in "${splitTag_list[@]}"; do
    for init_mode in "${init_mode_list[@]}"; do
      for peft_mode in "${peft_mode_list[@]}"; do
        
        # --- Create a short, suitable job name ---
        if [ "${peft_mode}" == "adapter" ]; then peft_short="adp"; else peft_short="lora"; fi
        init_short="${init_mode,,}"
        job_name="${peft_short}-${init_short}-${dsID}-${splitTag}"

        expID="init"

        # --- Construct and execute the command ---
        
        # 1. Build the command in an array for safe execution
        cmd=(
            sbatch 
            -J "${job_name}" 
            traintest_singlegpu_AlexPlasmaCells.sh 
            --expID "${expID}"
            --dsID "${dsID}" 
            --splitTag "${splitTag}" 
            --init_mode "${init_mode}"
            --peft_mode "${peft_mode}"
        )

        # 2. Print which job is being submitted for user feedback
        echo "Submitting Job: ${job_name}"
        echo "   Command -> ${cmd[*]}" # Show the full command being run

        # 3. Execute the command to submit the job
        "${cmd[@]}"
        
        echo "---"

        # 4. Pause for a second to avoid overwhelming the scheduler
        sleep 1

      done
    done
  done
done

echo "All jobs have been submitted."