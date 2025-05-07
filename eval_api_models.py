import os
import openai
import json
import httpx
import requests
import random
from joblib import Parallel, delayed
from typing import List, Dict
from dotenv import load_dotenv
from tqdm.auto import tqdm
from collections import defaultdict

def create_array_of_json_tasks(prompts, model="gpt-4o", temperature=0.0, logprobs=5, max_tokens=4, seed=24):
    tasks = []

    for index, prompt in enumerate(prompts):
        
        messages = prompt_to_messages(prompt)

        task = {
            "custom_id": f"prompt-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "logprobs": logprobs,
                "max_tokens": max_tokens,
                "seed": seed,
                "messages": messages,
            }
        }
        
        tasks.append(task)
    
    return tasks

def save_jsonl(tasks, file_name):
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')


def create_batch_file(client, file_name):
    batch_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )
    return batch_file

def create_batch_job(client, batch_file):
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch_job

def prompt_to_messages(prompt):
    """Assumes that the prompt is in the format:
    <definition>\n\n<demo1>\n\n<demo2>\n\n...<demoN>\n\n<test_sample>
    """
    definition, *demos, test_sample = prompt.split("\n\n")
    definition = definition + " PAY ATTENTION TO THE OUTPUT FORMAT -- ONLY OUTPUT THE ANSWER WITHOUT ANY OTHER TEXT, LIKE IN EXAMPLES."    
    
    messages = [
        {
            "role": "system",
            "content": definition,
        },
        {
            "role": "user",
            "content": "\n\n".join(demos)
        },
        {
            "role": "user",
            "content": test_sample
        }
    ]
    return messages


def openrouter_api_call(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, top_logprobs: int, seed: int):
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=messages,
        # provider={
        #     "order": ["Chutes"],
        # },
        # allow_fallback=False,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        logprobs=top_logprobs > 0,
        top_logprobs=top_logprobs,
        seed=seed
    )
    return completion


def local_api_call(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, top_logprobs: int, seed: int):
    client = openai.OpenAI(
        base_url = "http://192.168.24.161:30050/v1",
        api_key="EMPTY"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=top_logprobs > 0,
        top_logprobs=top_logprobs,
        seed=seed,
    )
    pass
    return completion.choices[0].message.content


def openrouter_request_call(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, top_logprobs: int, seed: int):
    url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": top_logprobs > 0,
        "top_logprobs": top_logprobs,
        "seed": seed,
        "provider": {
            "order": ["Chutes"]
        },
        "allow_fallback": False
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers).json()

    print(response.keys())
    print(response["error"])
    print(response["user_id"])

    return response["choices"][0]["message"]["content"]

def get_batch_majority_vote_predictions(format_to_generations: List[List[str]], tie_break_seed: int):
    ensemble_predictions = []

    n_formats = len(format_to_generations)
    n_samples = len(format_to_generations[0])

    for sample_index in range(n_samples):
        individual_generations = [format_to_generations[i][sample_index] for i in range(n_formats)]
        ensemble_predictions.append(get_majority_vote_candidates(individual_generations))

    # Break ties with random choice
    state = random.getstate()
    random.seed(tie_break_seed)
    ensemble_predictions = [random.choice(p) if len(p) > 1 else p[0] for p in ensemble_predictions]
    random.setstate(state)
    
    return ensemble_predictions

def _normalize_generation(generation: str):
    return generation.strip(' .,()\n-><').lower()

def get_majority_vote_candidates(individual_generations: List[str]):
    normalized_generations = [_normalize_generation(g) for g in individual_generations]

    generation_to_count = defaultdict(int)

    for generation in normalized_generations:
        generation_to_count[generation] += 1

    top_count = max(generation_to_count.values())
    top_generations = [generation for generation, count in generation_to_count.items() if count == top_count]

    return top_generations


def main():
    load_dotenv()

    # model = "deepseek/deepseek-chat-v3-0324:free"
    # model = "qwen/qwen3-235b-a22b:free"           # switches into thinking mode automatically
    # model = "meta-llama/llama-4-maverick:free"    # returns token ids instead of text tokens with top_logprobs
    # model = "google/gemma-3-27b-it:free"          # does not support developer role in messages
    # model = "qwen/qwen-2.5-72b-instruct:free"       # returns token ids instead of text tokens with top_logprobs
    model = "Qwen/Qwen2.5-72B-Instruct"
    task = "task320"
    ensemble_id = 0
    temperature = 0.0
    top_logprobs = 5
    max_tokens = 4
    seed = 24
    input_path = f"formatted_prompts/input_prompt_string_list_{task}_ensemble_id_{ensemble_id}.json"
    answer_path = f"formatted_prompts/input_prompt_string_list_{task}_answers.json"
    output_path = f"exp/{model}_{task}_ensemble_id_{ensemble_id}.json"

    with open(input_path, "r") as f:
        formatted_queries: List[List[str]] = json.load(f)
    
    n_formats = len(formatted_queries)
    n_samples = len(formatted_queries[0])

    print(f"Evaluating {n_formats} formats with {n_samples} samples each")

    format_to_generations = []

    for format_index, format_prompts in enumerate(formatted_queries):
        all_messages = [prompt_to_messages(p) for p in format_prompts]
        generations = Parallel(n_jobs=16)(
            delayed(local_api_call)(m, model, max_tokens, temperature, top_logprobs, seed)
            for m in tqdm(all_messages, desc=f"Processing format {format_index}")
        )
        format_to_generations.append(generations)

    ensemble_predictions = get_batch_majority_vote_predictions(format_to_generations, seed + ensemble_id)

    with open(answer_path, "r") as f:
        answers: List[str] = json.load(f)

    assert len(ensemble_predictions) == len(answers), f"Number of predictions {len(ensemble_predictions)} does not match number of answers {len(answers)}"

    accuracy = sum(1 for pred, answer in zip(ensemble_predictions, answers) if pred == _normalize_generation(answer)) / len(answers)

    # transport = httpx.SyncProxyTransport.from_url(f'socks5://{os.environ.get("PROXY_USER")}:{os.environ.get('PROXY_PASS')}@193.124.46.176:8080')
    # http_client = httpx.Client(transport=transport)

    # client = openai.OpenAI(
    #     http_client=http_client,
    #     api_key=api_key,
    # )

    # generations = Parallel(n_jobs=16)(
    #     delayed(openrouter_request_call)(messages, model, max_tokens, temperature, top_logprobs, seed)
    #     for messages in all_messages
    # )

    # generations = [
    #     openrouter_request_call(messages, model, max_tokens, temperature, top_logprobs, seed)
    #     for messages in all_messages
    # ]


    with open(output_path, "w") as f:
        result = {
            "model": model,
            "temperature": temperature,
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "seed": seed,
            "ensemble_id": ensemble_id,
            "input_path": input_path,
            "accuracy": accuracy,
            "ensemble_predictions": ensemble_predictions,
            "generations": generations,
        }
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()