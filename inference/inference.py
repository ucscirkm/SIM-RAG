#No retrieval version for triviaQA
import argparse
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_template import *
import re
import json
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import math
from accelerate import Accelerator
from utils.evaluation import *
from utils.text_processing import safe_literal_eval, extract_final_answer_and_rationale, extract_assistant_output, parse_query
import ast
import torch
import openai

def call_reasoner_batch(system_prefix, task_contents, gpt=False, temperature=0.6,top_p=0.9):
    # Prepare messages for all batches
    messages = [
        [
            {"role": "system", "content": system_prefix},
            {"role": "user", "content": task_content} 
        ] for task_content in task_contents # Prepare messages for all batches
    ]
    if gpt:
        responses = []
        for message in messages:
        
            # Make API call to OpenAI for batch processing
            response = openai.chat.completions.create(
                model=MODEL,  # Specify the GPT-4 model
                messages=message,
                temperature=temperature,
                max_tokens=256,  
                stop=None
            )

            responses.append(response.choices[0].message.content)
        return responses
    else:
        messages = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(messages, padding="longest", return_tensors="pt")
        #inputs = {key: val.to(model.device) for key, val in inputs.items()} # Cache the V
        inputs = {key: val.to(accelerator.device) for key, val in inputs.items()}  # Move to correct device
        
        # Batch generation
        outputs_batch = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],  # Apply the attention mask as well
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        # Batch decoding
        responses = tokenizer.batch_decode(
            outputs_batch, 
            skip_special_tokens=True
        )

        # Extract the assistant's output from each response
        extracted_outputs = [extract_assistant_output(response) for response in responses]
        
        return extracted_outputs

def call_search_batch(queries, all_past_docs, k=2, remove_repeat_docs=True):
    """
    Call bm25_search_batch and ensure results exclude documents in all_past_docs if remove_repeat_docs is True.
    """
    titles, retrieved_texts = bm25_search_batch(queries, all_past_docs, k, remove_repeat_docs)
    for index, (title, retrieved_text) in enumerate(zip(titles, retrieved_texts)):
        # If no results were found for the query
        if title is None or len(title) < k:
            titles[index] = ["No results found."] * k
        if retrieved_text is None or len(retrieved_text) < k:
            retrieved_texts[index] = ["No results found."] * k
    return titles, retrieved_texts

def bm25_search_batch(query_texts, all_past_docs, k=2, remove_repeat_docs=True):
    """
    Perform BM25 search, optionally excluding documents present in all_past_docs.
    """
    results_batch = []
    titles_batch = []
    
    # Ensure all_past_docs is a list of sets for efficient lookups
    if remove_repeat_docs:
        all_past_docs = [set(past_docs) for past_docs in all_past_docs]

    for query_idx, query_text in enumerate(query_texts):
        query = {"1": query_text}
        results = retriever.retrieve(corpus, query)
        query_id, scores_dict = list(results.items())[0]
        scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
        
        if remove_repeat_docs:
            # Filter out documents already in all_past_docs[query_idx]
            filtered_scores = [
                (doc_id, score) for doc_id, score in scores
                if corpus[doc_id].get('title', "No results found.") not in all_past_docs[query_idx]
            ]
        else:
            # Use unfiltered scores
            filtered_scores = scores
        
        # Retrieve top `k` documents
        top_texts = [corpus[filtered_scores[i][0]].get('text', "No results found.") 
                     for i in range(min(k, len(filtered_scores)))]
        top_titles = [corpus[filtered_scores[i][0]].get('title', "No results found.") 
                      for i in range(min(k, len(filtered_scores)))]
        
        # Append results
        results_batch.append(top_texts)
        titles_batch.append(top_titles)

    return titles_batch, results_batch


def update_task_with_retrieved_text(task_content, search_query, retrieved_text):
    updated_content = task_content + "Query: " + search_query + "\nRetrieved Document: " + retrieved_text + "\n"
    return updated_content

def save_results_to_csv(df, filename):
    """Helper function to save DataFrame to CSV."""
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def call_gate_batch(answers, rationales, task_contents, gate_model, gate_tokenizer, weighted_training):
    instruction = "Instruction: Predict if the following answer to the question and context should be accepted, 1, or rejected, 0, based on the rationale."
    gate_inputs = [
        f"{instruction}\n{task_content} \nAnswer: {answer}\nRationale: {rationale}"
        for answer, rationale, task_content in zip(answers, rationales, task_contents)
        ]

    # to accelerator device
    inputs = gate_tokenizer(gate_inputs, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
    
    if weighted_training:
        generated_ids = gate_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1  # only need one new token
        )
        # responses = gate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        responses = gate_tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)
        # Get logits
        last_token_logits = generated_ids.scores[0]  # shape: [batch_size, vocab_size]
        # Apply softmax for probs
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
        
        # Get token id for "0" and "1" 
        token_0_id = gate_tokenizer.encode("0", add_special_tokens=False)[0]
        token_1_id = gate_tokenizer.encode("1", add_special_tokens=False)[0]
        
        # Get probs
        probs_0 = probs[:, token_0_id].cpu().numpy()
        probs_1 = probs[:, token_1_id].cpu().numpy()
        
        predictions = [response == '1' for response in responses]
        confidences = {
            'predictions': predictions,
            'confidence_0': probs_0,
            'confidence_1': probs_1
        }
        return confidences 
    else:
        generated_ids = gate_model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )

        responses = gate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [response == '1' for response in responses]

def main_batch(task_contents, question_type, ids, batch_gold_answers, gold_retrieved_docs, checkpoint_file, max_turns, weighted_training, gpt, search_query_setting, top_docs, remove_repeat_docs):
    cur = 0
    results = []
    all_past_docs = [[] for _ in task_contents]

    if search_query_setting == 'multi_ex':
        search_query_prompt_chosen = search_query_prompt_multi_ex
    elif search_query_setting == 'single_ex':
        search_query_prompt_chosen = search_query_prompt_single_ex
    elif search_query_setting == '2wiki':
        search_query_prompt_chosen = search_query_prompt_2wiki
    else:
        search_query_prompt_chosen = search_query_prompt

    while cur <= max_turns:
        if gpt:
            if cur == 0:
                batch_responses = call_reasoner_batch(abstain_force_answer_prompt, task_contents, gpt, 0.6)
            else:
                batch_responses = call_reasoner_batch(force_answer_prompt, task_contents, gpt, 0.6)
        else:
            batch_responses = call_reasoner_batch(force_answer_prompt, task_contents, gpt, 0.6)
        print(f"REASONER ANSWERS: {batch_responses}")
        answers, rationales = zip(*[extract_final_answer_and_rationale(response, question_type) for response in batch_responses])

        verdicts = call_gate_batch(answers, rationales, task_contents, gate_model, gate_tokenizer, weighted_training)
        if weighted_training:
            # Modified to include confidence scores
            confidence_0 = verdicts['confidence_0']  # confidence for 0
            confidence_1 = verdicts['confidence_1']  # confidence for 1
            verdicts = verdicts['predictions']  # Temp solution to prevent bug - overwrite verdicts to match previous format
        
        if cur == max_turns:
            search_queries_parsed = [''] * len(task_contents)
            retrieved_titles = [[] for _ in task_contents]  # Initialize empty lists for multiple docs
            retrieved_texts_parsed = [[] for _ in task_contents]
        else:
            # Add retrieval
            search_query_responses = call_reasoner_batch(search_query_prompt_chosen, task_contents, gpt, 0.6)
            search_queries_parsed = [parse_query(response) for response in search_query_responses]
            print(f"PARSED REASONER QUERIES: {search_queries_parsed}")

            retrieved_titles, retrieved_texts_parsed = call_search_batch(search_queries_parsed, all_past_docs, top_docs, remove_repeat_docs)
            for i, title_list in enumerate(retrieved_titles):
                all_past_docs[i].extend(title_list)
            print(f"RETRIEVED TEXTS: {retrieved_texts_parsed}")
        
        remaining_task_indices = []
        for i, (answer, rationale, verdict, query, retrieved_title_list, retrieved_text_list) in enumerate(zip(answers, rationales, verdicts, search_queries_parsed, retrieved_titles, retrieved_texts_parsed)):
            # Combine multiple documents into a single string (if necessary for downstream reasoning)
            retrieved_title_combined = " | ".join(retrieved_title_list)
            retrieved_text_combined = " | ".join(retrieved_text_list)

            result_entry = {
                'ID': ids[i],
                'Turn': cur,
                'Reasoner Task Content': task_contents[i],
                'Reasoner Answer': normalize_answer(answer),
                'Reasoner Rationale': rationale,
                'Critic Output': verdict,
                'Search Query': query,
                'Retrieved Titles': retrieved_title_list,  # Keep as list for clarity
                'Retrieved Docs': retrieved_text_list,    # Keep as list for clarity
                'Gold Answer': batch_gold_answers[i],
                'Gold Retrieved Docs': gold_retrieved_docs[i],
                'Correct Answer': exact_match_multiple(answer, batch_gold_answers[i]),
                'Correct Retrieval': any(exact_match_multiple(title, gold_retrieved_docs[i]) for title in retrieved_title_list)
            }
            # Add weighted training-specific fields if enabled
            if weighted_training:
                result_entry['Confidence Scores'] = f"0: {confidence_0[i]:.4f}, 1: {confidence_1[i]:.4f}"
            results.append(result_entry)

            if not verdict:
                remaining_task_indices.append(i)

            # Update the task content with all retrieved documents
            retrieved_docs_content = "\n".join([f"Title: {title} Content: {text}" for title, text in zip(retrieved_title_list, retrieved_text_list)])
            task_contents[i] = update_task_with_retrieved_text(task_contents[i], query, retrieved_docs_content)

        # Update the remaining tasks
        task_contents = [task_contents[i] for i in remaining_task_indices]
        ids = [ids[i] for i in remaining_task_indices]
        answers = [answers[i] for i in remaining_task_indices]
        rationales = [rationales[i] for i in remaining_task_indices]
        verdicts = [verdicts[i] for i in remaining_task_indices]
        batch_gold_answers = [batch_gold_answers[i] for i in remaining_task_indices]
        gold_retrieved_docs = [gold_retrieved_docs[i] for i in remaining_task_indices]

        # If all tasks are accepted, break the loop
        if len(task_contents) == 0:
            break

        cur += 1

    # Save results to CSV at the end of processing
    save_df = pd.DataFrame(results)
    save_results_to_csv(save_df, checkpoint_file)

    # Extract final answers in batch
    return results

def run_system(input_file, output_file, question_type, log_file, batch_size=8, max_turns=4, weighted_training=False, gpt=False,search_query_setting='default',top_docs=2, remove_repeat_docs=False):
    start_time = time.time()

    original_data = pd.read_csv(input_file)

    for i in range(0,len(original_data), batch_size): 
        batch = original_data.iloc[i:i + batch_size]
        batch_ids = [entry['ID'] for _, entry in batch.iterrows()]
        batch_questions = [entry['Question'] for _, entry in batch.iterrows()]
        batch_gold_answers = [safe_literal_eval(entry['Answers']) for _, entry in batch.iterrows()]
        if gpt:
            task_contents = [f"Question: {entry['Question']}\n" for _, entry in batch.iterrows()]
        else:
            task_contents = [f"Question: {entry['Question']}\nContext:\n" for _, entry in batch.iterrows()]
        gold_retrieved_docs = [safe_literal_eval(entry['Documents']) for _,entry in batch.iterrows()]

        batch_predicted_answers = main_batch(
            ids=batch_ids,
            task_contents=task_contents,
            question_type=question_type,
            batch_gold_answers=batch_gold_answers,
            gold_retrieved_docs=gold_retrieved_docs,
            checkpoint_file = output_file,
            max_turns = max_turns,
            weighted_training=weighted_training,
            gpt=gpt,
            search_query_setting=search_query_setting,
            top_docs=top_docs,
            remove_repeat_docs=remove_repeat_docs
        )
        batch_start = i
        print(f"Processed batch {batch_start // batch_size + 1}/{(len(original_data) + batch_size - 1) // batch_size}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Generation complete. Results saved to CSV.")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    df = pd.read_csv(output_file)
    results= custom_evaluation(df, "Total")
    last_entries = df.groupby('ID').tail(1)
    results2 = custom_evaluation(last_entries, "Last Datapoint")

    with open(log_file, "a") as f:
        f.write(f"\n===== Evaluation Results for Total =====\n")
        for metric, value in results.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            elif isinstance(value, int):
                f.write(f"{metric}: {value}\n")
            elif value is None:
                f.write(f"{metric}: No valid entries\n")
            else:
                f.write(f"{metric}\n")
        f.write("===================================================\n")

    with open(log_file, "a") as f:
        f.write(f"\n===== Evaluation Results for Last Turn =====\n")
        for metric, value in results2.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            elif isinstance(value, int):
                f.write(f"{metric}: {value}\n")
            elif value is None:
                f.write(f"{metric}: No valid entries\n")
            else:
                f.write(f"{metric}\n")

    original_stdout = sys.stdout
    with open(log_file, "a") as f:
        sys.stdout = f
        try:
            sys.stdout = f
            f.write("=======================Evaluation Per Turn============================\n")
            evaluate_per_turn(df)
            f.write("===================================================\n")
        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Run the inference system with the Critic with various parameters to get predictions on validation data.')
    
    # Define command-line arguments
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment, used for naming and paths.')
    parser.add_argument('--dm_path', type=str, help='Path/directory of the Critic model. Default is generated using the experiment name.')
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--output_path', type=str, required=False, help='Path to the output dataset CSV file. Default is generated using the experiment name.')
    parser.add_argument('--log_path', type=str, required=False, help='Path to the log file. Default is generated using the experiment name.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to process at a time.')
    parser.add_argument('--max_turns', type=int, default=4,help='Max turns of responses, starting at 0.')
    parser.add_argument('--question_type', type=str, choices=['MCQ', 'OEQ','MATH'], default='OEQ', help='Type of question: MCQ for multiple choice, OEQ for open-ended, MATH for open-ended math.')
    parser.add_argument('--weighted_training', action="store_true", default=False, help='Boolean flag to indicate whether to use weighted training.')
    parser.add_argument('--gpt', action="store_true", default=False, help='Use llama (default) or GPT if pass in True.')
    parser.add_argument('--search_query_setting', type=str, choices=['multi_ex', 'single_ex','2wiki','default'], default='default', help='Type of search query prompt: default, single_ex, or multi_ex.')
    parser.add_argument('--top_docs', type=int, default=2, help='Num top documents retrieved and used.')
    parser.add_argument('--remove_repeat_docs', action="store_true", default=False, help='Removes repeated docs if True.')
    parser.add_argument('--wiki_corpus', action="store_true", default=False, help='Use 2wikisearchcorpus if True.')

    # Parse the arguments
    args = parser.parse_args()

    # Set default for input_path based on experiment_name if not provided
    if args.input_path is None:
        args.input_path = f'data/original/{args.experiment_name}_dev.csv'
    if args.output_path is None:
        args.output_path = f'predictions/{args.experiment_name}_predictions.csv'
    if args.log_path is None:
        args.log_path = f'logs/{args.experiment_name}_log.txt'
    if args.dm_path is None:
        args.dm_path = f'dm_training/finetuned_checkpoints/{args.experiment_name}'

    # Initialize the accelerator (as before)
    accelerator = Accelerator()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    #print(torch.cuda.is_available())
    torch.cuda.empty_cache()

    ## Search ##
    import pickle
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.lexical import BM25Search as BM25

    if args.wiki_corpus:
        # Load corpus and retriever settings
        with open('bm25_search/corpus/2wiki_corpus.pkl', 'rb') as f:
            corpus = pickle.load(f)
        with open('bm25_search/corpus/2wiki_retriever_settings.pkl', 'rb') as f:
            settings = pickle.load(f)
    else:
        # Load corpus and retriever settings
        with open('bm25_search/corpus/corpus.pkl', 'rb') as f:
            corpus = pickle.load(f)
        with open('bm25_search/corpus/retriever_settings.pkl', 'rb') as f:
            settings = pickle.load(f)
        
    index_name = settings['index_name']
    hostname = settings['hostname']
    number_of_shards = settings['number_of_shards']
    initialize = False  # Set to False as the index is already created

    retrieval_model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
    retriever = EvaluateRetrieval(retrieval_model)


    ## LLM ##
    from transformers import AutoTokenizer, AutoModelForCausalLM
    #import torch
    import openai
    
    if args.gpt:
        # Ensure OpenAI API key is provided
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided via environment variable OPENAI_API_KEY.")
        openai.api_key = openai_api_key
        MODEL = args.gpt_model if args.gpt_model else "gpt-4o-mini-2024-07-18"
    else:
        model_id = os.getenv("LLAMA_PATH")
        if not model_id:
            raise ValueError("Llama model path must be provided via environment variable LLAMA_PATH.")
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer.pad_token = tokenizer.eos_token

    ## GATE ###
    from transformers import AutoModelForSeq2SeqLM

    gate_model_id=args.dm_path
    gate_model_id_tok =args.dm_path + "_tokenizer"
    gate_tokenizer = AutoTokenizer.from_pretrained(gate_model_id_tok)
    gate_model = AutoModelForSeq2SeqLM.from_pretrained(
        gate_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Spread layers across GPUs
        )

    if args.gpt:
        gate_model, gate_tokenizer = accelerator.prepare(gate_model, gate_tokenizer)
    else:
        model, gate_model, tokenizer, gate_tokenizer = accelerator.prepare(model, gate_model, tokenizer, gate_tokenizer)

    # Call the run_system function with parsed arguments
    run_system(args.input_path, args.output_path, args.question_type, args.log_path, args.batch_size, args.max_turns, args.weighted_training, args.gpt, args.search_query_setting, args.top_docs, args.remove_repeat_docs)
