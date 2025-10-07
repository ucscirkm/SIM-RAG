# SIM-RAG: Self-practicing for Inner Monologue-based Retrieval Augmented Generation

This is the repository for the paper: [Knowing You Don’t Know: Learning When to Continue Search in
Multi-round RAG through Self-Practicing](https://arxiv.org/abs/2505.02811) (SIGIR '25). It provides a framework to run SIM-RAG experiments and evaluate models. Follow the steps below to set up and run your experiments.

## Prerequisites

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your/repository.git
   cd repository
   ```

2. Ensure you have all the required dependencies installed (refer to `requirements.txt` or installation instructions in the repo).

3. If you're using GPT, make sure to set your API key in the environment. You can do this by adding the following line to your .bashrc, .zshrc, or equivalent shell configuration file:

    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

    Then, run:
     ```bash
    source ~/.bashrc  # or `source ~/.zshrc` for Zsh users
    ```

4. Likewise, if you're using Llama, make sure to set the local path to Llama in the environment:
    
    ```bash
    export LLAMA_PATH="/path/to/your/llama"
    ```

    Then, run:
     ```bash
    source ~/.bashrc  # or `source ~/.zshrc` for Zsh users
    ```

## Prepare Data
1. Download our prebuilt corpus files `corpus.pkl`, `wiki_corpus.pkl`, `retriever_settings.pkl`, and `wiki_retriever_settings.pkl` into the `bm25_search` directory for retrieval.

```bash
git clone https://huggingface.co/datasets/dyang39/SIM-RAG-Corpus bm25_search
```

2. (Optional) Prepare the original datasets.

The datasets have already been prepared and are ready to use. However, if you'd like to prepare them yourself, you can place the 2WikiMultihopQA dataset (downloaded from its [GitHub repository](https://github.com/Alab-NII/2wikimultihop)) in the `data` directory, and the scripts will automatically load HotpotQA and TriviaQA directly from Hugging Face. Once ready, run the following scripts to process the datasets:
 
```bash
python /data/prepare_2wikimultihopqa.py
python /data/prepare_triviaqa.py
python /data/prepare_hotpotqa.py
```

## Run Experiment

To run the SIM-RAG experiment, you'll first need to create a custom script using `run_SIM-RAG.py`. This script will guide you through entering parameters and generating an executable `.sh` file for your experiment.

1. Run `run_SIM-RAG.py`:
   ```bash
   python run_SIM-RAG.py
   ```

2. Follow the prompts to enter details for the SIM-RAG experiment.

3. After you have entered all details, the script will generate a `.sh` file in the `bash_scripts` directory, which can be used to run the SIM-RAG experiment.

4. Change the permissions of the generated `.sh` file to make it executable:
   ```bash
   chmod +x bash_scripts/{script_filename}
   ```

5. Run the generated `.sh` file to start the experiment:
   ```bash
   ./bash_scripts/{script_filename}
   ```

## Evaluating Predictions

Once the SIM-RAG experiment is complete, you can evaluate the predictions using `evaluate_SIM-RAG.py`.

1. The predictions for each dataset are saved in the `predictions/` directory in the format `{name}_predictions.csv`.

2. To evaluate the predictions, run `evaluate_SIM-RAG.py` with the experiment ame:
   ```bash
   python evaluate_SIM-RAG.py --experiment_name {name}
   ```

3. The script will output the EM and F1 score of the predictions.

Fine-grained, intermediate evaluation data and statistics can also be found in `logs/{name}_log.txt`

## Customizing Experiments

To run offline, modify the `.sh` file to run each command with nohup. If you have downloaded checkpoints for an already trained Critic, modify the `.sh` to only run the last line (inference) while passing the path to the Critic into `--dm_path`. Make sure the tokenizer is in the same directory.

### Critic Checkpoints

We provide a general-purpose Critic:

1. [SIM-RAG-Llama3-2B](https://huggingface.co/dyang39/SIM-RAG-Llama3-2B): This Flan-T5-based Critic is fine-tuned on six datasets, including TriviaQA, HotPotQA, 2WikiMultiHopQA, PopQA, and Musique. It can be directly used for a general-purpose Critic in our Inference pipeline.



To reproduce our result in the paper, we provide the following checkpoints:

2. [SIM-RAG-GPT4-2B](https://huggingface.co/dyang39/SIM-RAG-GPT4-2B)

3. [SIM-RAG-Llama3-2B-hotpotqa](https://huggingface.co/dyang39/SIM-RAG-Llama3-2B-hotpotqa)

5. [SIM-RAG-Llama3-783M](https://huggingface.co/dyang39/SIM-RAG-783M)

## Citation 
If you find this work useful, please kindly cite 

```
@article{yang2025rag,
  title={Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing},
  author={Yang, Diji and Zeng, Linda and Rao, Jinmeng and Zhang, Yi},
  journal={arXiv preprint arXiv:2505.02811},
  year={2025}
}
```
