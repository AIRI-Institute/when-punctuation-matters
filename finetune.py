import ast
import json
import torch
import torch.nn.functional as F
import random
import pandas as pd
import wandb
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, List, Any
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers.data.data_collator import DataCollatorMixin


from generate_test_formats import (
    VANILLA_MAPPING_ALL_CATEGORIES,
    COMPOSITIONAL_TRAIN_SEPARATOR_LIST,
    COMPOSITIONAL_TRAIN_SPACE_LIST,
    TASK_NAMES,
    _get_format_hash
)
from utils import INSTRUCTION_PART_TAG, RESPONSE_PART_TAG

def load_model(name: str, max_seq_length: int, dtype: type, load_in_4bit: bool, device: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model.to(device)

    return model, tokenizer


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def augment_conversation(conversation, format_split_mode: str, test_format_hashes: Set[int] | None) -> Dict[str, str]:
    """Apply a random training prompt format to a conversation.
    """
    mapping_all_categories = {key: value for key, value in VANILLA_MAPPING_ALL_CATEGORIES.items()}
    if format_split_mode == "compositional_separator_space":
        mapping_all_categories["chosen_space"] = [(e, e) for e in COMPOSITIONAL_TRAIN_SPACE_LIST]
        mapping_all_categories["chosen_separator"] = [(e, e) for e in COMPOSITIONAL_TRAIN_SEPARATOR_LIST]
    elif format_split_mode == "random":
        pass
    else:
        raise NotImplementedError("Modes other than 'compositional_separator_space' and 'random' are not supported.")

    verbalizer_first_options = ["question", "query", "Q"]
    verbalizer_second_options = ["answer", "reply", "A"]

    filtered_conversation = [msg for msg in conversation if msg["from"] in ("human", "gpt")]
    filtered_conversation = filtered_conversation[:2]
    assert filtered_conversation[0]["from"] == "human"
    assert filtered_conversation[1]["from"] == "gpt"

    # For "random" format_split_mode, sample until we don't get a format which is not in test set:
    while True:
        _text_descriptor_fn, _text_descriptor_fn_str = random.choice(mapping_all_categories["text_descriptor_fn"])
        _separator, _serarator_str = random.choice(mapping_all_categories["chosen_separator"])
        if "\n" in _separator:
            _space, _space_str = random.choice([e for e in mapping_all_categories["chosen_space"] if "\n" in e[0]])
        else:
            _space, _space_str = random.choice(mapping_all_categories["chosen_space"])

        current_hash = _get_format_hash(_space_str, _serarator_str, _text_descriptor_fn_str)

        if format_split_mode != "random" or current_hash not in test_format_hashes:
            break
        else:
            pass
            # print(f"Regenerating format", [_space, _separator, _text_descriptor_fn_str])

    _text_descriptor_first = _text_descriptor_fn(random.choice(verbalizer_first_options))
    _text_descriptor_second = _text_descriptor_fn(random.choice(verbalizer_second_options))
    text = f"{INSTRUCTION_PART_TAG}"\
            f"{_text_descriptor_first}{_separator}{filtered_conversation[0]['value']}{_space}"\
            f"{_text_descriptor_second}{_separator}"\
            f"{RESPONSE_PART_TAG}"\
            f"{filtered_conversation[1]['value']}"

    return {"text": text}

def apply_default_format(conversation):
    """Apply the default format to a conversation.
    """
    filtered_conversation = [msg for msg in conversation if msg["from"] in ("human", "gpt")]
    filtered_conversation = filtered_conversation[:2]
    assert filtered_conversation[0]["from"] == "human"
    assert filtered_conversation[1]["from"] == "gpt"

    text = f"{INSTRUCTION_PART_TAG}"\
            f"Question: {filtered_conversation[0]['value']}\n"\
            f"Answer: {RESPONSE_PART_TAG}{filtered_conversation[1]['value']}"

    return {"text": text}

def _compose_test_hashes_set(format_split_mode: str, path_to_test_formats: str) -> Set[int] | None:
    """If split mode is "random", computes hashes for test prompt formats.
    Resulting set is used to efficiently check whether a random generated format matches with some test format.
    """
    if format_split_mode == "random":
        config = load_json(path_to_test_formats)
        
        test_format_hashes = set()
        for format in config["test_formats"]:
            space_idx = config["action_types"].index("chosen_space")
            sep_idx = config["action_types"].index("chosen_separator")
            text_descriptor_fn_idx = config["action_types"].index("text_descriptor_fn")

            test_format_hashes.add(
                _get_format_hash(format[space_idx], format[sep_idx], format[text_descriptor_fn_idx])
            )
    else:
        # For all other modes, leakage is avoided in a different way,
        # through patching the set of possible choices for format elements.
        test_format_hashes = None

    return test_format_hashes


def _build_natural_instruction_dataset(path_to_test_formats: str, start_idx: int) -> Dict[str, List[Dict[str, str]]]:
    instances = []

    for taskname in tqdm(TASK_NAMES, desc="tasks"):
        task_path = list(Path("../natural-instructions/tasks").glob(f"{taskname}_*.json"))
        assert len(task_path) == 1, f"Found more than one file for {taskname},\n{task_path}"
        task_path = task_path[0]

        current_task_path = path_to_test_formats.replace("task050", taskname)
        with open(current_task_path, "r") as f:
            config = json.load(f)

        # amount of demonstrations was originally hardcoded in data_loading.py, set_up_prompt_variation_exploration_without_extra_files function. Must be changed synchronously.
        n_demonstrations = 10
        max_train_size = 1000
        train_ids = config["dataset_ordered_ids"][start_idx + n_demonstrations:start_idx + n_demonstrations + max_train_size]

        task = load_json(task_path)
        instances.extend([task["Instances"][i] for i in train_ids])
    
    dataset = {"conversations": []}
    for instance in tqdm(instances, "instances to conversations"):
        conversation = [
            {
                "from": "human",
                "value": instance["input"]
            },
            {
                "from": "gpt",
                "value": instance["output"][0]
            }
        ]
        dataset["conversations"].append(conversation)

    return dataset


def get_dataset(dataset_name: str, n_samples: int, n_augmentations: int, format_split_mode: str, path_to_test_formats: str, start_idx: int, loss_type: str) -> Dataset:
    """Load a dataset.
    Args:
        dataset_name: path to .csv file or a huggingface dataset name.
        n_samples: size of a subsample for huggingface datasets.
        n_augmentations: for each example from (a subsample of) original dataset, `n_augmentation` versions of it with different formats will be added to the final dataset.
        format_split_mode: defines which augmentations can be applied -- we can only use formats from train split.
        path_to_test_formats: for format_split_mode = 'random', we need a path to a complete list test formats.
        start_idx: if 'dataset_name' == "natural-instructions", we need to specify the starting index of training samples, since some samples are taken as test.
    Output:
        final_dataset: dataset with augmented versions of original samples.
    """
    seed = 23
    random.seed(seed)

    print(dataset_name)
    if dataset_name.endswith(".csv"):
        dataset = pd.read_csv(dataset_name)
        dataset["conversations"] = dataset["conversations"].map(ast.literal_eval)
    elif dataset_name == "natural-instructions":
        dataset = _build_natural_instruction_dataset(path_to_test_formats, start_idx)
        random.shuffle(dataset["conversations"])
    else:
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    test_format_hashes = _compose_test_hashes_set(format_split_mode, path_to_test_formats)

    all_formatted_texts = []
    for conversation in dataset["conversations"]:
        for _ in range(n_augmentations):
            # When training with consistency loss, we need the first example of each pair to be the default format.
            if loss_type == "consistency":
                all_formatted_texts.append(apply_default_format(conversation))
            all_formatted_texts.append(augment_conversation(conversation, format_split_mode, test_format_hashes))

    final_dataset = Dataset.from_list(all_formatted_texts)

    print(f"Original sample conversation: {dataset['conversations'][0]}")
    print(f"Augmented example, v1: {final_dataset[0]['text']}")
    print(f"Augmented example, v2: {final_dataset[1]['text']}")

    return final_dataset


def prepare_dataset(dataset): 
    clean = [x for x in dataset['text'][::2]]
    perturb = [x for x in dataset['text'][1::2]]

    return Dataset.from_dict(
        {
            "text": clean, 
            "perturbed_text": perturb
        }
    )

def custom_tokenizer(examples, instructions_pattern, response_pattern, tokenizer, max_length):

    texts_c = [x.split(response_pattern)[0].replace(instructions_pattern, '') for x in examples["text"]]
    labels_c = [x.split(response_pattern)[1] for x in examples["text"]]

    texts_p = [x.split(response_pattern)[0].replace(instructions_pattern, '') for x in examples["perturbed_text"]]
    labels_p = [x.split(response_pattern)[1] for x in examples["perturbed_text"]]


    clean_encodings = tokenizer(
        text = texts_c,
        text_target = labels_c,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        add_special_tokens = False
    )

    perturb_encodings = tokenizer(
        text=texts_p,
        text_target=labels_p,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        add_special_tokens = False
    )

    def _mask_non_predicted(labels, input_ids):
        padded_labels = []
        n_preds = []

        for i in range(len(labels)):
            n_pred = sum(input_ids[i] != tokenizer.pad_token_id).item()
            left_pad = torch.ones(n_pred, dtype=labels[0].dtype) * tokenizer.pad_token_id
            padded_label = torch.cat((left_pad, labels[i][:-n_pred]))
            padded_labels.append(padded_label)
            n_preds.append(n_pred)

        return padded_labels, n_preds

    masked_labels_c, len_mask_c = _mask_non_predicted(clean_encodings["labels"],clean_encodings["input_ids"])
    masked_labels_p, len_mask_p = _mask_non_predicted(perturb_encodings["labels"],perturb_encodings["input_ids"])

    return {
        "input_ids_c": clean_encodings["input_ids"],
        "labels_c": masked_labels_c,
        "label_mask_len_c": len_mask_c,
        "attention_mask_c": clean_encodings["attention_mask"],
        "input_ids_p": perturb_encodings["input_ids"],
        "labels_p": masked_labels_p,
        "label_mask_len_p": len_mask_p,
        "attention_mask_p": perturb_encodings["attention_mask"]

    }


@dataclass
class LoraArguments:
    target_modules: Tuple
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    bias: str
    use_gradient_checkpointing: bool | str
    random_state: int
    use_rslora: bool = False


@dataclass
class DatasetArguments:
    dataset_name: str
    n_original_samples: int
    n_augmentations: int
    format_split_mode: str
    path_to_test_formats: str
    start_idx: int


class CustomDataCollator(DataCollatorMixin):
    """
    Custom Data Collator to prepare batched inputs required for consistency loss computation.
    """

    def __init__(self, *args, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        input_ids_c = [torch.tensor(f["input_ids_c"]) for f in features]
        label_mask_len_c = [torch.tensor(f["label_mask_len_c"]) for f in features]
        labels_c = [torch.tensor(f["labels_c"]) for f in features]
        attention_mask_c = [torch.tensor(f["attention_mask_c"]) for f in features]

        input_ids_p = [torch.tensor(f["input_ids_p"]) for f in features]
        label_mask_len_p = [torch.tensor(f["label_mask_len_p"]) for f in features]
        labels_p = [torch.tensor(f["labels_p"]) for f in features]
        attention_mask_p = [torch.tensor(f["attention_mask_p"]) for f in features]

        batch = {
            "input_ids_c": torch.stack(input_ids_c),
            "label_mask_len_c": torch.stack(label_mask_len_c),
            "labels_c": torch.stack(labels_c),
            "attention_mask_c": torch.stack(attention_mask_c),

            "input_ids_p": torch.stack(input_ids_p),
            "label_mask_len_p": torch.stack(label_mask_len_p),
            "labels_p": torch.stack(labels_p),
            "attention_mask_p": torch.stack(attention_mask_p)
        }

        return batch

# custom class for JS-loss
class CustomSFTTrainer(SFTTrainer):

    def __init__(self, *args, loss_type='consistency', beta=10, **kwargs):
        """
        Added the additional property 'loss_type' during initialization.
        """
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.beta = beta

        if self.loss_type == 'consistency':
            self.compute_loss = self._compute_loss_consistency

        print(f"At init: {self.args.report_to=}")

    def _kl_divergence(self, p, q):
        return F.kl_div(p.log(), q, reduction='batchmean') 

    def _jensen_shannon_divergence(self, p, q):
        m = 0.5 * (p + q)  
        return 0.5 * (self._kl_divergence(p, m) + self._kl_divergence(q, m))
    

    def _perturbation_consistency_loss(self, y_c, denom_c, y_p, denom_p):
        y_c_avg = y_c.sum(dim=1) / (2048 - denom_c.requires_grad_(False)).unsqueeze(-1) #TODO: change to max_length
        y_p_avg = y_p.sum(dim=1) / (2048 - denom_p.requires_grad_(False)).unsqueeze(-1)

        loss_js = self._jensen_shannon_divergence(y_c_avg, y_p_avg).mean()
        return loss_js

    def _compute_loss_consistency(self, model, inputs, *args, **kwargs):
        input_ids_c = inputs["input_ids_c"]
        labels_c = inputs["labels_c"]
        label_mask_len_c = inputs['label_mask_len_c']
        attention_mask_c = inputs["attention_mask_c"]

        input_ids_p = inputs["input_ids_p"]
        labels_p = inputs["labels_p"]
        label_mask_len_p = inputs['label_mask_len_p']
        attention_mask_p = inputs["attention_mask_p"]

        outputs_c = model(input_ids=input_ids_c, attention_mask=attention_mask_c)
        logits_c = outputs_c.logits

        outputs_p = model(input_ids=input_ids_p, attention_mask=attention_mask_p)
        logits_p = outputs_p.logits

        loss_mask_c = (labels_c != self.tokenizer.pad_token_id) 
        loss_mask_p = (labels_p != self.tokenizer.pad_token_id)

        lc = F.cross_entropy(
            logits_c.view(-1, logits_c.shape[-1]),
            labels_c.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction="mean"
        )

        lp = F.cross_entropy(
            logits_p.view(-1, logits_p.shape[-1]),
            labels_p.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            reduction="mean"
        )

        y_c_prob = F.softmax(logits_c, dim=-1)
        y_p_prob = F.softmax(logits_p, dim=-1)

        l_js = self._perturbation_consistency_loss(
            y_c_prob * loss_mask_c.unsqueeze(-1),
            label_mask_len_c, 
            y_p_prob * loss_mask_p.unsqueeze(-1),
            label_mask_len_p
        )

        autoscale = (lc.detach() + lp.detach()) / (l_js.detach() + 1e-4)

        total_loss = lc + lp + self.beta * autoscale * l_js

        # Log individual loss components
        if "wandb" in self.args.report_to and self.state.global_step % self.args.logging_steps == 0:
            print("Logging to wandb")
            wandb.log(
                {
                    "loss/cross_entropy_first": lc.item(),
                    "loss/cross_entropy_second": lp.item(),
                    "loss/perturbation_consistency": l_js.item(),
                    "loss/total": total_loss.item()
                },
                step=self.state.global_step
            )

        return total_loss


def run_finetuning(model_name: str, lora_arguments: LoraArguments, training_arguments: TrainingArguments, dataset_arguments: DatasetArguments,
                   seed: int, max_seq_length: int, dtype: torch.dtype, load_in_4bit: bool, device: str, output_dir: str, loss_type: str, beta: float):
    model, tokenizer = load_model(model_name, max_seq_length, dtype, load_in_4bit, device)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_arguments.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=lora_arguments.target_modules,
        lora_alpha=lora_arguments.lora_alpha,
        lora_dropout=lora_arguments.lora_dropout,      # Supports any, but = 0 is optimized
        bias=lora_arguments.bias,                      # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing=lora_arguments.use_gradient_checkpointing, # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=lora_arguments.use_rslora,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ
    )

    dataset = get_dataset(
        dataset_arguments.dataset_name,
        dataset_arguments.n_original_samples,
        dataset_arguments.n_augmentations,
        dataset_arguments.format_split_mode,
        dataset_arguments.path_to_test_formats,
        dataset_arguments.start_idx,
        loss_type
    )

    if loss_type != 'consistency':
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=2,
            packing=False, # Can make training 5x faster for short sequences.
            args=training_arguments
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_PART_TAG,
            response_part=RESPONSE_PART_TAG,
        )
        
        print("Check masking")
        print(RESPONSE_PART_TAG, tokenizer.encode(RESPONSE_PART_TAG), [tokenizer.decode(tok) for tok in tokenizer.encode(RESPONSE_PART_TAG)])
        print("ORIGINAL:", tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
        space = tokenizer(" ", add_special_tokens = False).input_ids[0]
        print("MASKED:", tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))
        
    else:     
        prepared_dataset = prepare_dataset(dataset)
        
        dataset = prepared_dataset.map(
            lambda x: custom_tokenizer(x, INSTRUCTION_PART_TAG, RESPONSE_PART_TAG, tokenizer, max_seq_length),
            batched=True,
            batch_size=1024,
            num_proc=4
        )
        
        trainer = CustomSFTTrainer(
            model=model,
            train_dataset=dataset,
            data_collator=CustomDataCollator(tokenizer=tokenizer),
            args=training_arguments, 
            loss_type=loss_type,
            beta=beta
        )


    #Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer_stats = trainer.train()

    show_training_stats(trainer_stats, start_gpu_memory, max_memory)

    # Save model
    model_name = model_name.split("/")[-1]
    save_path = f"{output_dir}/{model_name}_lora"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model, tokenizer


def show_training_stats(trainer_stats, start_gpu_memory, max_memory):
    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


def sample_inference(model, tokenizer):
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    query = "Question: Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,\nAnswer: "
    inputs = tokenizer(query, return_tensors="pt").to("cuda:0").input_ids

    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True,
                            temperature=1.5, min_p=0.1)
    generation = tokenizer.batch_decode(outputs)

    print(f"{generation=}")


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed", type=int, default=3023)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Dataset arguments
    parser.add_argument("--n-original-samples", type=int, default=8000)
    parser.add_argument("--n-augmentations", type=int, default=4)
    parser.add_argument("-d", "--dataset-name", default="teknium/OpenHermes-2.5")
    parser.add_argument("--format-split-mode", required=True)
    parser.add_argument("--path-to-test-formats")
    parser.add_argument("--start-idx", type=int, default=0)

    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-rslora", action="store_true")

    # Training arguments
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("--loss-type", type=str, default='default')
    parser.add_argument("--beta", type=float, default=10.0)

    # Wandb arguments
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="prompt-instability", help="Wandb project name")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)

    dtype = torch.bfloat16

    lora_arguments = LoraArguments(
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=args.use_rslora,
    )

    dataset_arguments = DatasetArguments(
        dataset_name=args.dataset_name,
        n_original_samples=args.n_original_samples,
        n_augmentations=args.n_augmentations,
        format_split_mode=args.format_split_mode,
        path_to_test_formats=args.path_to_test_formats,
        start_idx=args.start_idx
    )
    if args.loss_type == "consistency": 
        remove_unused_columns = False
    else: 
        remove_unused_columns = True

    # Configure wandb reporting
    report_to = "none"
    if args.use_wandb:
        report_to = "wandb"
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
        print(f"Wandb initialized: {wandb.run.name}")

    training_arguments = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        # max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=report_to,  # Use wandb if enabled
        save_total_limit=3,
        save_steps=100,
        remove_unused_columns=remove_unused_columns
    )
    

    model, tokenizer = run_finetuning(args.model_name, lora_arguments, training_arguments, dataset_arguments, args.seed,
                                      args.max_seq_length, dtype, args.load_in_4bit, args.device, args.output_dir, args.loss_type, args.beta)

    sample_inference(model, tokenizer)

    # Close wandb run if it was used
    if args.use_wandb and 'wandb' in locals():
        wandb.finish()