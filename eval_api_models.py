import httpx
import json
import logging
import os
import openai
import requests
import random
import time
from collections import defaultdict, Counter
from datetime import datetime
from dotenv import load_dotenv
from httpx_socks import SyncProxyTransport
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import List, Dict, Optional
from statistics import stdev, median

######### Constants #########

TASKS = [
    "task1423",     # mathqa, geometry
    "task1419",     # mathqa, gain
    "task1420",     # mathqa, general
    "task322",      # does the sentence contain a threat
    "task320",      # race stereotypes
    "task323",      # is the sentence sexually explicit
    "task296",      # story completion
    "task1387",     # anli, entailment
    "task161",      # count number of words containing letter X
    "task114",      # is X the longest word in the sentence
]

MULTIPLE_CHOICE_TASKS = [
    "task1423",     # mathqa, geometry
    "task1419",     # mathqa, gain
    "task1420",     # mathqa, general
    "task296",      # story completion
]


######### Json utils #########

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_jsonl(tasks, file_name):
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def read_jsonl(file_name):
    with open(file_name, 'r') as file:
        return [json.loads(line) for line in file]


######### OpenAI batch API utils #########

def create_array_of_json_requests(prompts, model="gpt-4o", temperature=0.0, logprobs=5, max_tokens=4, seed=24):
    requests = []

    for index, prompt in enumerate(prompts):
        
        messages = prompt_to_messages(prompt)

        request = {
            "custom_id": f"prompt-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "logprobs": logprobs > 0,
                "top_logprobs": logprobs,
                "max_tokens": max_tokens,
                "seed": seed,
                "messages": messages,
            }
        }
        
        requests.append(request)
    
    return requests

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


def launch_batches_for_task(client: openai.OpenAI, task: str, model: str, max_tokens: int, temperature: float, top_logprobs: int, seed: int, n_jobs: int, force_new_batch_files: bool = False):
    batch_job_ids = {}
    print(f"{task=}")
    for ensemble_id in range(10):
        print(f"\t{ensemble_id=}")
        input_path = f"formatted_prompts/input_prompt_string_list_{task}_ensemble_id_{ensemble_id}.json"
        if not os.path.exists(input_path):
            logging.warning(f"Input file {input_path} does not exist. Skipping.")
            continue
        
        formatted_ensemble_queries: List[List[str]] = load_json(input_path)
        # If more than 5 formats, use the last 5 to ensure consistent ensemble size
        formatted_ensemble_queries = formatted_ensemble_queries[-5:]
        if len(formatted_ensemble_queries) != 5:
            logging.warning(f"Expected 5 formats per ensemble, got {len(formatted_ensemble_queries)} for {input_path}")
        
        if not os.path.exists("batch_input_files"):
            os.makedirs("batch_input_files")

        batch_job_ids[f"ensemble_id_{ensemble_id}"] = {}

        for format_index, formatted_queries in enumerate(formatted_ensemble_queries):
            batch_file_name = f"batch_input_files/batch_file_{task}_ensemble_id_{ensemble_id}_format_{format_index}.jsonl"
            
            if not os.path.exists(batch_file_name) or force_new_batch_files:
                requests = create_array_of_json_requests(formatted_queries, model, temperature, top_logprobs, max_tokens, seed)
                save_jsonl(requests, batch_file_name)

            batch_file = create_batch_file(client, batch_file_name)
            # confirmation = input(f"Launch {task} {ensemble_id=} {format_index=}? (y/n)")
            # if confirmation == "y":
            batch_job = create_batch_job(client, batch_file)
            print(f"\t\t{format_index=} launched")
            batch_job_ids[f"ensemble_id_{ensemble_id}"][f"format_id_{format_index}"] = batch_job.id

    return batch_job_ids

def main_batch_openai(tasks: List[str]):
    load_dotenv()

    transport = SyncProxyTransport.from_url(f"socks5://{os.environ.get('PROXY_USER')}:{os.environ.get('PROXY_PASS')}@193.124.46.176:8080")
    http_client = httpx.Client(transport=transport)

    client = openai.OpenAI(
        http_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    model = "gpt-4.1-2025-04-14"
    temperature = 0.0
    top_logprobs = 10
    max_tokens = 4
    seed = 24
    n_jobs = 128

    per_task_batch_job_ids = {}

    per_task_batch_job_ids["config"] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_logprobs": top_logprobs,
        "seed": seed,
        "n_jobs": n_jobs,
    }

    for task in tasks:
        batch_job_ids = launch_batches_for_task(client, task, model, max_tokens, temperature, top_logprobs, seed, n_jobs)
        per_task_batch_job_ids[task] = batch_job_ids

    save_json("per_task_batch_job_ids.json", per_task_batch_job_ids)


def watch_batch_jobs(batch_file_name: Optional[str] = None):
    load_dotenv()
    transport = SyncProxyTransport.from_url(f"socks5://{os.environ.get('PROXY_USER')}:{os.environ.get('PROXY_PASS')}@193.124.46.176:8080")
    http_client = httpx.Client(transport=transport)

    client = openai.OpenAI(
        http_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    if batch_file_name is None:
        batches = client.batches.list()
        # print(f"{len(list(batches))=}")
        status_counts = Counter(batch.status for batch in batches)
        print(status_counts)
        print(sum(status_counts.values()), " batches total")
        return

    per_task_batch_job_ids = load_json(batch_file_name)
    per_task_batch_job_ids.pop("config")
    for task, ensemble_to_format_to_batch_job_ids in per_task_batch_job_ids.items():
        print(task)

        def _retrieve_job(ensemble_id, format_id, batch_job_id):
            batch_job = client.batches.retrieve(batch_job_id)
            return ensemble_id, format_id, batch_job

        batch_jobs = Parallel(n_jobs=16, backend="threading")(delayed(_retrieve_job)(ensemble_id, format_id, batch_job_id)
            for ensemble_id, format_to_batch_job_ids in ensemble_to_format_to_batch_job_ids.items()
            for format_id, batch_job_id in format_to_batch_job_ids.items()
        )

        current_ensemble_id = -1
        for ensemble_id, format_id, batch_job in batch_jobs:
            if ensemble_id != current_ensemble_id:
                current_ensemble_id = ensemble_id
                print(f"\t{ensemble_id}")

            suffix = "(has error file)" if batch_job.error_file_id is not None else ""

            print(f"\t\t{format_id}: {batch_job.status} {suffix}")


def cancel_batch_jobs(batch_file_name: str):
    load_dotenv()
    transport = SyncProxyTransport.from_url(f"socks5://{os.environ.get('PROXY_USER')}:{os.environ.get('PROXY_PASS')}@193.124.46.176:8080")
    http_client = httpx.Client(transport=transport)

    client = openai.OpenAI(
        http_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    per_task_batch_job_ids = load_json(batch_file_name)
    per_task_batch_job_ids.pop("config")
    for task, ensemble_to_format_to_batch_job_ids in per_task_batch_job_ids.items():
        print(task)
        for ensemble_id, format_to_batch_job_ids in ensemble_to_format_to_batch_job_ids.items():
            print(f"\t{ensemble_id}")
            for format_id, batch_job_id in format_to_batch_job_ids.items():
                batch_job = client.batches.retrieve(batch_job_id)
                if batch_job.status != "completed":
                    confirmation = input(f"Cancel {batch_job_id}? (y/n)")
                    if confirmation == "y":
                        client.batches.cancel(batch_job_id)


def save_results_from_batch_jobs(overwrite_batch_files: bool = False):
    load_dotenv()
    transport = SyncProxyTransport.from_url(f"socks5://{os.environ.get('PROXY_USER')}:{os.environ.get('PROXY_PASS')}@193.124.46.176:8080")
    http_client = httpx.Client(transport=transport)

    client = openai.OpenAI(
        http_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    per_task_batch_job_ids = load_json("per_task_batch_job_ids.json")
    per_task_batch_job_ids.pop("config")

    for task, ensemble_to_format_to_batch_job_ids in per_task_batch_job_ids.items():
        print(task)
        for ensemble_id, format_to_batch_job_ids in ensemble_to_format_to_batch_job_ids.items():
            print(f"\t{ensemble_id}")
            for format_id, batch_job_id in format_to_batch_job_ids.items():
                batch_job = client.batches.retrieve(batch_job_id)
                print(f"\t\t{format_id}: {batch_job.status}")

                if batch_job.status == "completed":
                    if batch_job.error_file_id is not None:
                        # error_file_id = batch_job.error_file_id
                        # error_file = client.files.content(error_file_id).content
                        # with open("error_file.json", "w") as f:
                        #     print(error_file, file=f)

                        logging.warning(f"Error file {batch_job.error_file_id} for {task} {ensemble_id=} {format_id=}, skipping")
                        continue
        
                    content = client.files.content(batch_job.output_file_id).content
                    result_file_name = f"batch_output_files/{task}_{ensemble_id}_{format_id}.jsonl"
                    if not os.path.exists(result_file_name) or overwrite_batch_files:
                        logging.info(f"Saving batch file {result_file_name}")
                        with open(result_file_name, 'wb') as file:
                            file.write(content)
                    else:
                        logging.warning(f"Batch file {result_file_name} already exists. Skipping.")

def compute_metrics_openai(task: str, is_multiple_choice: bool, ensemble_amount: int = 10, ensemble_size: int = 5, few_shot_generations_amount: int = 10):
    
    config = load_json("per_task_batch_job_ids.json").pop("config")

    answer_path = f"formatted_prompts/input_prompt_string_list_{task}_answers.json"
    answers: List[str] = load_json(answer_path)
    answers = [_normalize_generation(answer, is_multiple_choice) for answer in answers]

    ensemble_accuracies = []
    few_shot_generations: List[List[str]] = []

    for ensemble_id in range(ensemble_amount):
        output_path = f"exp/{config['model']}/{task}/ensemble_id_{ensemble_id}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if os.path.exists(output_path):
            logging.warning(f"Output file {output_path} already exists. Skipping.")
            result = load_json(output_path)
            ensemble_accuracies.append(result["accuracy"])
            if len(few_shot_generations) < few_shot_generations_amount:
                few_shot_generations.append(result["generations"])
            continue
        
        ensemble_id_to_generations: List[List[str]] = []
        source_batch_output_files = []

        for format_id in range(ensemble_size):
            result_file_name = f"batch_output_files/{task}_ensemble_id_{ensemble_id}_format_id_{format_id}.jsonl"
            if not os.path.exists(result_file_name):
                logging.warning(f"Batch file {result_file_name} does not exist. Skipping.")
                continue

            outputs = read_jsonl(result_file_name)
            generations: List[str] = [
                output["response"]["body"]["choices"][0]["message"]["content"]
                for output in outputs
            ]
            ensemble_id_to_generations.append(generations)
            assert len(generations) == len(answers), f"Number of predictions {len(generations)} does not match number of answers {len(answers)} for {ensemble_id=}, {format_id=}"

            source_batch_output_files.append(result_file_name)

            if len(few_shot_generations) < few_shot_generations_amount:
                few_shot_generations.append(generations)

        ensemble_predictions = get_batch_majority_vote_predictions(ensemble_id_to_generations, config["seed"] + ensemble_id, is_multiple_choice=task in multiple_choice_tasks)
        assert len(ensemble_predictions) == len(answers), f"Number of predictions {len(ensemble_predictions)} does not match number of answers {len(answers)} for {answer_path}"

        accuracy = sum(1 for pred, answer in zip(ensemble_predictions, answers) if pred == answer) / len(answers)
        ensemble_accuracies.append(accuracy)
        result = {
            "task": task,
            "model": config["model"],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "seed": config["seed"],
            "ensemble_id": ensemble_id,
            "source_batch_output_files": source_batch_output_files,
            "accuracy": accuracy,
            "ensemble_predictions": ensemble_predictions,
            "generations": generations,
            "top_logprobs": config["top_logprobs"],
            "n_jobs": config["n_jobs"],
        }
        save_json(output_path, result)

    # Non-ensemble results
    accuracies = []
    for generation in few_shot_generations[:few_shot_generations_amount]:
        if len(generation) != len(answers):
            logging.warning(f"Number of predictions {len(generation)} does not match number of answers {len(answers)} for {task}")
        accuracy = sum(1 for pred, answer in zip(generation, answers) 
                        if answer == _normalize_generation(pred, is_multiple_choice)
        ) / len(answers)
        accuracies.append(accuracy)

    result = {
        "task": task,
        "model": config["model"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "seed": config["seed"],
        "n_jobs": config["n_jobs"],
        "spread": max(accuracies) - min(accuracies),
        "std_accuracy": stdev(accuracies),
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "median_accuracy": median(accuracies),
        "ensemble_spread": max(ensemble_accuracies) - min(ensemble_accuracies),
        "ensemble_std_accuracy": stdev(ensemble_accuracies),
        "ensemble_mean_accuracy": sum(ensemble_accuracies) / len(ensemble_accuracies),
        "ensemble_median_accuracy": median(ensemble_accuracies),
        "accuracies": accuracies,
        "ensemble_accuracies": ensemble_accuracies,
        "top_logprobs": config["top_logprobs"],
    }
    output_path = f"exp/{config['model']}/{task}/summary.json"
    save_json(output_path, result)

    
######### Prompt utils #########

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
            "content": "\n\n".join(demos) + "\n\n" + test_sample
        },
    ]
    return messages

######### API calls utils #########

def _compute_delay(attempt: int, base_delay: float) -> float:
    """Compute exponential backoff delay with jitter.
    
    Args:
        attempt: Current retry attempt (0-based)
        base_delay: Base delay in seconds
    
    Returns:
        Delay in seconds before next retry
    """
    return (base_delay * (2 ** attempt)) + random.uniform(0, 0.1 * base_delay)

def post_with_retries(url: str, payload: Dict, headers: Dict, max_retries: int = 3, base_delay: float = 0.1) -> Dict:
    """Make HTTP POST request with retry logic.
    
    Retries on:
    - HTTP 429 (Too Many Requests - Rate limit exceeded)
    - HTTP 503 (Service Unavailable - Server temporarily unable to handle request)
    - HTTP 502 (Bad Gateway - Server received invalid response from upstream)
    - HTTP 504 (Gateway Timeout - Server didn't receive timely response from upstream)
    - Connection errors, timeouts, and general request exceptions
    """
    retryable_codes = {429, 503, 502, 504}
    
    for attempt in range(max_retries - 1):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            if e.response.status_code not in retryable_codes:
                raise
        except (requests.ConnectionError, requests.Timeout, requests.RequestException):
            pass
        except json.JSONDecodeError:
            raise   # JSONDecodeError can't be fixed by retrying
            
        time.sleep(_compute_delay(attempt, base_delay))
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def openrouter_request_call(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, top_logprobs: int, seed: int, max_retries: int = 3, base_delay: float = 0.1):
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

    return post_with_retries(url, payload, headers, max_retries, base_delay)


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
    return completion.choices[0].message.content


def get_batch_majority_vote_predictions(format_to_generations: List[List[str]], tie_break_seed: int, is_multiple_choice: bool):
    """Returns a list of ensemble predictions, one per sample. Predictions are normalized with `_normalize_generation`.
    """
    ensemble_predictions = []

    n_formats = len(format_to_generations)
    n_samples = len(format_to_generations[0])

    for sample_index in range(n_samples):
        individual_generations = [format_to_generations[i][sample_index] for i in range(n_formats)]
        ensemble_predictions.append(get_majority_vote_candidates(individual_generations, is_multiple_choice))

    # Break ties with random choice
    state = random.getstate()
    random.seed(tie_break_seed)
    ensemble_predictions = [random.choice(p) if len(p) > 1 else p[0] for p in ensemble_predictions]
    random.setstate(state)
    
    return ensemble_predictions

def _unicode_escape_sequence_to_str(generation: str):
    """Convert a unicode escape sequence to a string.
    Example: "\u2164" -> "Ⅴ"
    """
    return chr(int(generation.lstrip('\\u'), 16))

def _normalize_generation(generation: str, is_multiple_choice: bool):
    """Normalize the generation by removing punctuation and whitespace.
    For multiple choice tasks, also convert latin letters and roman numerals to numbers.
    """
    normalized_generation = generation.split()[0].strip(' .,()\n-><').lower()

    if normalized_generation.startswith("\\u"):
        normalized_generation = _unicode_escape_sequence_to_str(normalized_generation)

    roman_to_number = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10", "xi": "11", "xii": "12"}
    fancy_roman_to_number = {chr(0x215F + i + 1).lower(): str(i + 1) for i in range(12)}
    latin_to_number = {chr(ord('a') + i): str(i + 1) for i in range(12)}

    if is_multiple_choice:
        if normalized_generation in roman_to_number:
            normalized_generation = roman_to_number[normalized_generation]
        elif normalized_generation in latin_to_number:
            normalized_generation = latin_to_number[normalized_generation]
        elif normalized_generation in fancy_roman_to_number:
            normalized_generation = fancy_roman_to_number[normalized_generation]
        elif normalized_generation.isdigit():
            # already normalized form
            pass
        else:
            print("Multiple choice normalization failed for", generation)

    return normalized_generation

def get_majority_vote_candidates(individual_generations: List[str], is_multiple_choice: bool):
    normalized_generations = [_normalize_generation(g, is_multiple_choice) for g in individual_generations]

    generation_to_count = defaultdict(int)

    for generation in normalized_generations:
        generation_to_count[generation] += 1

    top_count = max(generation_to_count.values())
    top_generations = [generation for generation, count in generation_to_count.items() if count == top_count]

    return top_generations


def get_ensemble_outputs(formatted_ensemble_queries, model, max_tokens, temperature, top_logprobs, seed, n_jobs):
    format_to_outputs = []

    for format_index, formatted_queries in enumerate(formatted_ensemble_queries):
        all_messages = [prompt_to_messages(p) for p in formatted_queries]
        outputs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(openrouter_request_call)(messages, model, max_tokens, temperature, top_logprobs, seed)
            for messages in tqdm(all_messages, desc=f"Processing format {format_index}")
        )
        format_to_outputs.append(outputs)

    return format_to_outputs
        

def get_openrouter_responses(
    task: str, 
    model: str,
    temperature: float,
    top_logprobs: int,
    max_tokens: int,
    seed: int,
    n_jobs: int,
    amount_of_ensembles: int = 10,
    ensemble_size: int = 5
):
    """
    Sends API requests and stores output for a SINGLE task.
    
    Args:
        task: Name of the task from NaturalInstructions, e.g. "task050"
        model: OpenRouter model name, e.g. "deepseek/deepseek-chat-v3-0324"
        temperature
        top_logprobs
        max_tokens
        seed: Seed used in api calls
        n_jobs: Number of parallel jobs used by joblib
        amount_of_ensembles: Number of ensembles to generate
        ensemble_size: Number of formats to use in a single ensemble
    """
    load_dotenv()
    
    # Set up logging
    base_task_dir = f"exp/{model}/{task}"
    os.makedirs(base_task_dir, exist_ok=True)
    os.makedirs(f"{base_task_dir}/full_outputs", exist_ok=True)
    
    now = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    log_file = os.path.join(base_task_dir, f"run_{now}.log")
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    # Save config
    config = {
        "model": model,
        "temperature": temperature,
        "top_logprobs": top_logprobs,
        "max_tokens": max_tokens,
        "seed": seed,
        "n_jobs": n_jobs
    }
    config_path = f"{base_task_dir}/full_outputs/config.json"
    save_json(config_path, config)
    
    # Process each ensemble
    for ensemble_id in range(amount_of_ensembles):
        input_path = f"formatted_prompts/input_prompt_string_list_{task}_ensemble_id_{ensemble_id}.json"
        raw_outputs_path = f"{base_task_dir}/full_outputs/ensemble_id_{ensemble_id}.jsonl"
        
        # Skip if output already exists
        if os.path.exists(raw_outputs_path):
            logging.warning(f"Skipping {raw_outputs_path} because it already exists")
            continue
        
        # Check if input exists
        if not os.path.exists(input_path):
            fail_output_path = f"{base_task_dir}/ensemble_id_{ensemble_id}_FAIL.json"
            result = {
                "error_message": f"Input file {input_path} does not exist",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "ensemble_id": ensemble_id,
                "input_path": input_path,
                "n_jobs": n_jobs,
                "top_logprobs": top_logprobs,
            }
            save_json(fail_output_path, result)
            logging.warning(f"Input file {input_path} does not exist. Skipping.")
            continue
        
        # Load and process formatted queries
        formatted_ensemble_queries = load_json(input_path)
        # If more than `ensemble_size` formats, use the last `ensemble_size`, since the first format might be the same across ensembles
        formatted_ensemble_queries = formatted_ensemble_queries[-ensemble_size:]
        if len(formatted_ensemble_queries) != ensemble_size:
            logging.warning(f"Expected {ensemble_size} formats per ensemble, got {len(formatted_ensemble_queries)} for {input_path}")
        
        # Get outputs from OpenRouter API
        try:
            format_to_outputs = get_ensemble_outputs(formatted_ensemble_queries, model, max_tokens, temperature, top_logprobs, seed, n_jobs)
            save_jsonl(format_to_outputs, raw_outputs_path)
            
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            # Handle API and JSON parsing errors
            fail_output_path = f"{base_task_dir}/full_outputs/ensemble_id_{ensemble_id}_FAIL.json"
            result = {
                "error_message": str(e),
                "error_type": type(e).__name__,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "ensemble_id": ensemble_id,
                "input_path": input_path,
                "n_jobs": n_jobs,
                "top_logprobs": top_logprobs,
            }
            save_json(fail_output_path, result)
            logging.error(f"Error processing ensemble {ensemble_id}: {str(e)}")
            continue
        except OSError as e:
            # Handle file system errors
            logging.error(f"File system error for ensemble {ensemble_id}: {str(e)}")
            continue


def compute_metrics_openrouter(
    task: str, 
    model: str, 
    is_multiple_choice: bool,
    amount_of_ensembles: int = 10,
    ensemble_size: int = 5,
    max_few_shot_generations: int = 10
):
    """
    Loads outputs and computes metrics for a SINGLE task.
    
    Args:
        task: Name of the task
        model: Model used for generation
        is_multiple_choice: Whether the task is multiple choice
        ensemble_size: Number of formats to use in ensemble (default: 5)
    """
    base_task_dir = f"exp/{model}/{task}"
    
    # Load config
    config_path = f"{base_task_dir}/full_outputs/config.json"
    if not os.path.exists(config_path):
        logging.error(f"Config file {config_path} does not exist")
        return
    
    config = load_json(config_path)
    seed = config["seed"]
    
    # Load and normalize answers
    answer_path = f"formatted_prompts/input_prompt_string_list_{task}_answers.json"
    if not os.path.exists(answer_path):
        logging.error(f"Answers file {answer_path} does not exist")
        return
    
    answers = load_json(answer_path)
    answers = [_normalize_generation(answer, is_multiple_choice=is_multiple_choice) for answer in answers]
    
    # Initialize list for ensemble accuracies
    ensemble_accuracies = []
    format_generations = []  # To collect generations for individual format metrics
    
    # Process each ensemble
    for ensemble_id in range(amount_of_ensembles):
        processed_output_path = f"{base_task_dir}/ensemble_id_{ensemble_id}.json"
        
        # Skip if output already exists and add its accuracy
        if os.path.exists(processed_output_path):
            result = load_json(processed_output_path)
            if "accuracy" in result:
                ensemble_accuracies.append(result["accuracy"])
                
                # Add generations to format_generations for first two ensembles
                if len(format_generations) < max_few_shot_generations and "generations" in result:
                    format_generations.extend(result["generations"])
                    
                logging.info(f"Loaded existing accuracy for ensemble {ensemble_id}: {result['accuracy']}")
                continue
        
        # Load outputs from file
        raw_outputs_path = f"{base_task_dir}/full_outputs/ensemble_id_{ensemble_id}.jsonl"
        if not os.path.exists(raw_outputs_path):
            logging.warning(f"Output file {raw_outputs_path} does not exist. Skipping ensemble {ensemble_id}.")
            continue

        try:
            format_to_outputs: List[List[Dict]] = read_jsonl(raw_outputs_path)
            format_to_generations = []
            for format_id, outputs_with_given_format in enumerate(format_to_outputs):
                format_to_generations.append([output["choices"][0]["message"]["content"]
                    for output in outputs_with_given_format])

            # Add to format_generations for first two ensembles
            if len(format_generations) < max_few_shot_generations:
                format_generations.extend(format_to_generations)
            
            # Use only the last 'ensemble_size' formats
            format_to_generations = format_to_generations[-ensemble_size:]
            
            # Calculate ensemble predictions
            ensemble_predictions = get_batch_majority_vote_predictions(
                format_to_generations, 
                seed + ensemble_id, 
                is_multiple_choice=is_multiple_choice
            )
            
            if len(ensemble_predictions) != len(answers):
                logging.warning(
                    f"Number of predictions {len(ensemble_predictions)} does not match "
                    f"number of answers {len(answers)} for ensemble {ensemble_id}"
                )
            
            # Calculate accuracy
            accuracy = sum(1 for pred, answer in zip(ensemble_predictions, answers) 
                          if pred == answer) / len(answers)
            
            # Save results
            result = {
                "model": model,
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "seed": seed,
                "ensemble_id": ensemble_id,
                "accuracy": accuracy,
                "ensemble_predictions": ensemble_predictions,
                "generations": format_to_generations,
                "top_logprobs": config["top_logprobs"],
                "n_jobs": config["n_jobs"],
            }
            save_json(processed_output_path, result)
            
            # Add to ensemble accuracies
            ensemble_accuracies.append(accuracy)
            
        except Exception as e:
            logging.error(f"Error computing metrics for ensemble {ensemble_id}: {str(e)}")
            continue
    
    # Calculate individual format metrics
    format_generations = format_generations[:max_few_shot_generations]
    
    accuracies = []
    for generation in format_generations:
        if len(generation) != len(answers):
            logging.warning(f"Number of predictions {len(generation)} does not match number of answers {len(answers)}")
        
        # Calculate accuracy for each format
        accuracy = sum(1 for pred, answer in zip(generation, answers) 
                      if answer == _normalize_generation(pred, is_multiple_choice=is_multiple_choice)
                      ) / len(answers)
        accuracies.append(accuracy)
    
    # Create summary
    if accuracies:
        result = {
            "model": model,
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "seed": seed,
            "n_jobs": config["n_jobs"],
            "spread": max(accuracies) - min(accuracies) if accuracies else 0,
            "std_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0,
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "median_accuracy": median(accuracies) if accuracies else 0,
            "ensemble_spread": max(ensemble_accuracies) - min(ensemble_accuracies) if ensemble_accuracies else 0,
            "ensemble_std_accuracy": stdev(ensemble_accuracies) if len(ensemble_accuracies) > 1 else 0,
            "ensemble_mean_accuracy": sum(ensemble_accuracies) / len(ensemble_accuracies) if ensemble_accuracies else 0,
            "ensemble_median_accuracy": median(ensemble_accuracies) if ensemble_accuracies else 0,
            "accuracies": accuracies,
            "ensemble_accuracies": ensemble_accuracies,
            "top_logprobs": config["top_logprobs"],
        }
        
        # Save summary
        summary_path = f"{base_task_dir}/summary.json"
        save_json(summary_path, result)
    else:
        logging.error(f"No accuracies computed for {task}. Cannot create summary.")


if __name__ == "__main__":
    tasks = TASKS
    print("Tasks:", tasks)

    user_input = input("Continue? (y/n)\n")
    if user_input != "y":
        exit()


    # for task in tasks:
    #     get_openrouter_responses(
    #         task, 
    #         model="deepseek/deepseek-chat-v3-0324", 
    #         temperature=0.0, 
    #         top_logprobs=10, 
    #         max_tokens=4, 
    #         seed=24, 
    #         n_jobs=128, 
    #         amount_of_ensembles=10,
    #         ensemble_size=5
    #     )
    
    # main_batch_openai(TASKS)
    # watch_batch_jobs("per_task_batch_job_ids.json")
    # watch_batch_jobs(batch_file_name="per_task_batch_job_ids_May14_16-59.json")
    # save_results_from_batch_jobs()

    for task in tasks:
        compute_metrics_openrouter(
            task, 
            model="deepseek/deepseek-chat-v3-0324", 
            is_multiple_choice=task in MULTIPLE_CHOICE_TASKS, 
            amount_of_ensembles=10,
            ensemble_size=5,
            max_few_shot_generations=10
        )
    
        compute_metrics_openai(
            task,
            is_multiple_choice=task in MULTIPLE_CHOICE_TASKS,
            ensemble_amount=10,
            ensemble_size=5,
            few_shot_generations_amount=10
        )
    