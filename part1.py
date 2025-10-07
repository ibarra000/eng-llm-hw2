#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 2: Import Libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import subprocess
import os
import tempfile
import json
from pathlib import Path
from IPython.display import clear_output

print("Libraries imported successfully")



# In[6]:


# ============================================================================
# Cell 3: Define Helper Functions
# ============================================================================


def load_model_and_tokenizer(model_name: str):
    """
    Load the pre-trained model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        model: The loaded language model
        tokenizer: The corresponding tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    
    model = model.to(device)
    
    print("Model and tokenizer loaded successfully")
    return model, tokenizer


def load_test_dataset():
    """
    Load the test dataset for Racket code generation.
    
    Returns:
        dataset: The test problems dataset (the 'train' split)
    """
    dataset = load_dataset(
        "nuprl/engineering-llm-systems",
        "mbpp-rkt-test-problems"
    )
    
    test_data = dataset['train']
    print(f"Loaded {len(test_data)} test problems")
    return test_data


def generate_completions(model, tokenizer, prompt, num_completions=5):
    """
    Generate multiple completions for a given prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt text
        num_completions: Number of completions to generate (default: 5)
    
    Returns:
        completions: List of generated code strings
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    completions = []
    
    for i in range(num_completions):
        print(f"  Generating completion {i+1}/{num_completions}...", end='\r')
        
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            top_p=0.95,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        completion = generated_text[len(prompt):]
        completions.append(completion)
    
    print(f"Generated {num_completions} completions")
    return completions


def save_generation_to_json(task_id, problem, prompt, completions, output_dir="completions"):
    """
    Save generation information to a JSON file.
    
    Args:
        task_id: Problem task ID
        problem: The problem dictionary
        prompt: The prompt used for generation
        completions: List of generated completions
        output_dir: Directory to save JSON files
    
    Returns:
        filepath: Path to the saved JSON file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generation_data = {
        'task_id': task_id,
        'description': problem['description'],
        'input_format': problem['input_format'],
        'output_format': problem['output_format'],
        'prompt': prompt,
        'completions': completions,
        'test_cases': problem['tests']
    }
    
    filepath = os.path.join(output_dir, f"task_{task_id}.json")
    with open(filepath, 'w') as f:
        json.dump(generation_data, f, indent=2)
    
    return filepath


def generate_and_save_all_completions(model, tokenizer, dataset, output_dir="completions"):
    """
    Generate completions for all problems and save to JSON files.
    Skips problems that already have saved completions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: The test dataset
        output_dir: Directory to save JSON files
    
    Returns:
        num_generated: Number of problems actually generated (not skipped)
    """
    total_problems = len(dataset)
    num_generated = 0
    num_skipped = 0
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating completions for {total_problems} problems")
    
    for idx, problem in enumerate(dataset):
        task_id = problem['task_id']
        
        filepath = os.path.join(output_dir, f"task_{task_id}.json")
        
        if os.path.exists(filepath):
            num_skipped += 1
            continue
        
        
        description = problem['description']
        input_format = problem['input_format']
        output_format = problem['output_format']
        
        prompt = f"""; {description}
; Input format: {input_format}
; Output format: {output_format}

#lang racket

"""
        
        completions = generate_completions(model, tokenizer, prompt, num_completions=5)
        
        filepath = save_generation_to_json(task_id, problem, prompt, completions, output_dir)
        print(f"Saved to {filepath}\n")
        num_generated += 1
    
    print(f"Generation complete!")
    print(f"  Generated: {num_generated} new problems")
    print(f"  Skipped: {num_skipped} existing problems")
    print(f"  Total files in '{output_dir}/': {num_generated + num_skipped}")
    
    return num_generated


def test_racket_completion(completion_string, test_cases, racket_path=None):
    """
    Tests a Racket code completion against a list of test cases.
    """
    if racket_path is None:
        racket_path = globals().get('RACKET_PATH', 'racket')
    
    lines = completion_string.strip().split('\n')
    non_lang_lines = [line for line in lines if not line.strip().startswith('#lang')]
    sanitized_completion = "#lang racket\n" + "\n".join(non_lang_lines)
    
    results = []
    
    with tempfile.NamedTemporaryFile(
        mode='w+', suffix='.rkt', delete=False, encoding='utf-8'
    ) as temp_file:
        temp_file.write(sanitized_completion)
        temp_file_path = temp_file.name
    
    try:
        for case in test_cases:
            test_input = case['input']
            expected_output = case['output']
            passed = False
            
            try:
                process = subprocess.run(
                    [racket_path, temp_file_path],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if process.returncode != 0:
                    actual_output = f"RUNTIME ERROR:\n{process.stderr.strip()}"
                else:
                    actual_output = process.stdout.strip()
                    passed = (actual_output == expected_output)
                    
            except subprocess.TimeoutExpired:
                actual_output = "EXECUTION TIMED OUT (5 seconds)"
            
            #if passed:
                #print("\n--- Passed ---")
                #print(f"  Input:    '{test_input}'")
                #print(f"  Expected: '{expected_output}'")
                #print(f"  Actual:   '{actual_output}'")
                #print("--------------------------")

            results.append({
                'passed': passed,
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output
            })
            
    finally:
        os.remove(temp_file_path)
        
    return results



def check_test_cases(test_results):
    """
    Check if all test cases passed.
    
    Args:
        test_results: List of test result dictionaries from test_racket_completion
    
    Returns:
        passed: Boolean indicating if all tests passed
    """
    return all(result['passed'] for result in test_results)


def evaluate_from_json(json_filepath):
    """
    Load a generation JSON file and evaluate all completions.
    
    Args:
        json_filepath: Path to the JSON file
    
    Returns:
        passed: Boolean indicating if at least one completion passed
        results: Dictionary with detailed results for each completion
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    prompt = data['prompt']
    completions = data['completions']
    test_cases = data['test_cases']
    
    results = {
        'task_id': data['task_id'],
        'completions_tested': [],
        'any_passed': False
    }
    
    for i, completion in enumerate(completions):
        full_code = prompt + completion
        
        test_results = test_racket_completion(full_code, test_cases)
        
        all_tests_passed = check_test_cases(test_results)
        
        completion_result = {
            'completion_index': i,
            'all_tests_passed': all_tests_passed,
            'test_details': test_results
        }
        
        if all_tests_passed:
            results['any_passed'] = True
        
        results['completions_tested'].append(completion_result)
    
    return results['any_passed'], results


def evaluate_all_completions(completions_dir="completions"):
    """
    Evaluate all generated completions from JSON files.
    
    Args:
        completions_dir: Directory containing JSON files
    
    Returns:
        pass_at_1: The pass@1 score
        all_results: List of detailed results for each problem
    """
    json_files = sorted(Path(completions_dir).glob("task_*.json"))
    total_problems = len(json_files)
    
    if total_problems == 0:
        print(f"No JSON files found in '{completions_dir}/'")
        return 0.0, []
    
    print(f"Evaluating {total_problems} problems from '{completions_dir}/'")
    
    problems_passed = 0
    all_results = []
    
    for idx, json_file in enumerate(json_files):
        print(f"[{idx+1}/{total_problems}] Evaluating {json_file.name}")
        
        passed, detailed_results = evaluate_from_json(json_file)
        
        if passed:
            problems_passed += 1
            print(f"PASSED\n")
        else:
            print(f"FAILED\n")
        
        all_results.append(detailed_results)
    
    # Calculate pass@1 score
    pass_at_1 = problems_passed / total_problems
    
    print(f"Problems passed: {problems_passed}/{total_problems}")
    print(f"Pass@1 Score: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    
    return pass_at_1, all_results


# In[3]:


# ============================================================================
# Cell 4: Configuration
# ============================================================================

# for the purposes of evaluating each model, we changed the model name and completions directory
# to match the model being evaluated
MODEL_NAME = "./models/exp1_lr5e-5_1epoch/epoch_1"
COMPLETIONS_DIR = "completions_exp1"

RACKET_PATH = os.path.expanduser("~/racket/bin/racket")

print(f"Configuration set:")
print(f"  Model: {MODEL_NAME}")
print(f"  Completions directory: {COMPLETIONS_DIR}")
print(f"  Racket path: {RACKET_PATH}")


# In[8]:


# ============================================================================
# Cell 5: Load Model and Tokenizer
# ============================================================================

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)


# In[4]:


# ============================================================================
# Cell 6: Load Test Dataset
# ============================================================================

test_dataset = load_test_dataset()


# In[10]:


# ============================================================================
# Cell 7: Generate Completions and Save to JSON (This will take time!)
# ============================================================================

print("Starting generation... This may take a while!\n")
num_generated = generate_and_save_all_completions(model, tokenizer, test_dataset, COMPLETIONS_DIR)

print(f"\nGenerated and saved completions for {num_generated} problems")


# In[7]:


# ============================================================================
# Cell 8: Evaluate Completions from JSON Files
# ============================================================================

print("Starting evaluation from saved JSON files...\n")
pass_at_1_score, evaluation_results = evaluate_all_completions(COMPLETIONS_DIR)


# In[9]:


# ============================================================================
# Cell 9: Display Results
# ============================================================================


print("FINAL SUMMARY")
print(f"Model: {MODEL_NAME}")
print(f"Total Problems: {len(evaluation_results)}")
print(f"Problems Passed: {sum(1 for r in evaluation_results if r['any_passed'])}")
print(f"Pass@1 Score: {pass_at_1_score:.4f} ({pass_at_1_score*100:.2f}%)")

