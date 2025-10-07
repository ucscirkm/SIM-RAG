import os

def generate_command_script(name, dataset_name, use_weighted_training=True, gpt=False, gpus="1,2,3", top_docs=2, remove_repeat_docs=True):
    weighted_flag = "--weighted_training" if use_weighted_training else ""
    gpt_flag = "--gpt" if gpt else ""
    remove_repeat_docs_flag = "--remove_repeat_docs" if remove_repeat_docs else ""
    wiki_corpus_flag =  ""
        
    # Search query setting based on dataset name
    if dataset_name == "hotpotqa":
        search_query_setting = "multi_ex"
        max_turns = 3
    elif dataset_name == "triviaqa":
        search_query_setting = "single_ex"
        max_turns = 2
    elif dataset_name == "2wikimultihopqa":
        search_query_setting = "2wiki"
        wiki_corpus_flag = "--wiki_corpus"
        max_turns = 3

    sh_content = f"""#!/bin/bash
# This script runs all commands with specified settings

mkdir -p nohup_logs

# Create a log file for commands and errors
log_file="nohup_logs/{name}_command_log.txt"

# Function to run a command and handle errors
run_command() {{
    local cmd="$1"
    echo "Running: $cmd" | tee -a "$log_file"  # Print the command being executed
    if ! bash -c "$cmd"; then
        echo "Error: Command failed - $cmd" | tee -a "$log_file"  # Log error to command log
        exit 1  # Exit immediately if a command fails
    fi
    echo "Finished: $cmd" | tee -a "$log_file"  # Print when the command has successfully finished
}}

# Running generation for each dataset

# Running generation
run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 generation/generation.py --experiment_name {name} --input_path data/original/{dataset_name}_train.csv --max_turns {max_turns} {weighted_flag} {gpt_flag} --search_query_setting {search_query_setting} --top_docs {top_docs} {remove_repeat_docs_flag} {wiki_corpus_flag}'

# Preparing training
run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 dm_training/prepare_training.py --experiment_name {name}'

# Running main training
run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 dm_training/main.py --experiment_name {name} {weighted_flag}'

# Running inference
run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 inference/inference.py --experiment_name {name} --input_path data/original/{dataset_name}_test.csv {weighted_flag} {gpt_flag} --search_query_setting {search_query_setting} --top_docs {top_docs} {remove_repeat_docs_flag} {wiki_corpus_flag}'
"""

    os.makedirs("./bash_scripts", exist_ok=True)
    # Write the content to a .sh file
    script_filename = f"run_{name}.sh"
    with open("bash_scripts/" + script_filename, "w") as file:
        file.write(sh_content)
        print()
    print(f"{script_filename} has been created in bash_scripts.")
    print()
    print(f'''Example usage:
    chmod +x bash_scripts/{script_filename}
    ./bash_scripts/{script_filename}''')
    print()
    print(f"Running and error commands will be saved to: nohup_logs/{name}_command_log.txt")
    print(f"Detailed evaluation statistics during generation, training, and inference will be saved to: logs/{name}_.txt")
    print(f"Predictions will be saved to: predictions/{name}_predictions.csv")

if __name__ == "__main__":
    # User-defined settings
    name = input("Enter your custom SIM-RAG experiment name i.e. 'SIM-RAGtest1': ")
    
    # Accepting a single dataset name
    dataset_name = input("Enter the dataset name (e.g., hotpotqa, triviaqa, 2wikimultihopqa): ").strip()

    gpt = input("Use GPT? (yes/no): ").strip().lower() == "yes"
    gpus = input("Enter CUDA visible devices (e.g., 1,2,3): ")

    use_weighted_training = input("Optional hyperparameters, enter nothing for the following inputs to use defaults:\nUse weighted training? (yes/no) (Default no): ").strip().lower() == "yes"
    top_docs = input("Enter the number of top docs to retrieve (Default 2): ").strip()
    remove_repeat_docs = input("Remove repeated retrieved docs? (yes/no) (Default yes): ").strip().lower() == "yes"

    if not top_docs:
        top_docs = 2
    else:
        top_docs = int(top_docs) 

    # Generate the command script
    generate_command_script(name, dataset_name, use_weighted_training, gpt, gpus, top_docs, remove_repeat_docs)
