import torch
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Tuple
from datasets import load_dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling

from generate_train_val_test_formats import VANILLA_MAPPING_ALL_CATEGORIES, COMPOSITIONAL_TRAIN_SEPARATOR_LIST, COMPOSITIONAL_TRAIN_SPACE_LIST
VANILLA_MAPPING_ALL_CATEGORIES["chosen_space"] = [(e, e) for e in COMPOSITIONAL_TRAIN_SPACE_LIST]
VANILLA_MAPPING_ALL_CATEGORIES["chosen_separator"] = [(e, e) for e in COMPOSITIONAL_TRAIN_SEPARATOR_LIST]


def load_model(name, max_seq_length, dtype, load_in_4bit, device):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model.to(device)

    return model, tokenizer


def get_hermes_dataset():
    seed = 23
    random.seed(seed)
    verbalizer_first = "question"
    verbalizer_second = "answer"

    def formatting_prompts_func(examples):
        # TODO: insert augmentations here
        # x fix random seed
        # - how to handle multiple-choice queries?
        batch_conversations = examples["conversations"]

        formatted_conversations = []
        for conversation in batch_conversations:
            filtered_conversation = [msg for msg in conversation if msg["from"] in ("human", "gpt")]
            filtered_conversation = filtered_conversation[:2]
            assert filtered_conversation[0]["from"] == "human"
            assert filtered_conversation[1]["from"] == "gpt"

            current_text_descriptor_fn = random.choice(VANILLA_MAPPING_ALL_CATEGORIES["text_descriptor_fn"])[0]
            _verbalizer_first = current_text_descriptor_fn(verbalizer_first)
            _verbalizer_second = current_text_descriptor_fn(verbalizer_second)
            _separator = random.choice(VANILLA_MAPPING_ALL_CATEGORIES["chosen_separator"])[0]
            if "\n" in _separator:
                _space = random.choice([e[0] for e in VANILLA_MAPPING_ALL_CATEGORIES["chosen_space"] if "\n" in e[0]])
            else:
                _space = random.choice(VANILLA_MAPPING_ALL_CATEGORIES["chosen_space"])[0]

            formatted_conversations.append(
                f"{_verbalizer_first}{_separator}{filtered_conversation[0]['value']}{_space}"\
                f"{_verbalizer_second}{_separator}{filtered_conversation[1]['value']}"
            )

        return {"text": formatted_conversations}

    dataset_name = "teknium/OpenHermes-2.5"
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.shuffle(seed)

    print(f"{dataset[5]['conversations']=}")
    print(f"{dataset[5]['text']=}")

    return dataset


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


def run_finetuning(name: str, lora_arguments: LoraArguments, training_arguments: TrainingArguments, seed: int,
                   max_seq_length: int, dtype: torch.dtype, load_in_4bit: bool, device: str, output_dir: str):
    model, tokenizer = load_model(name, max_seq_length, dtype, load_in_4bit, device)

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

    dataset = get_hermes_dataset()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        dataset_num_proc=2,
        packing=False, # Can make training 5x faster for short sequences.
        args=training_arguments
    )

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    #     response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    # )

    # print(f"{tokenizer.decode(trainer.train_dataset[5]['input_ids'])=}")
    # space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    # print(f"{tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]['labels']])=}")

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
    model_name = name.split("/")[-1]
    save_path = f"{output_dir}/{model_name}_lora"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # model.push_to_hub("your_name/lora_model", token = "...")
    # tokenizer.push_to_hub("your_name/lora_model", token = "...")

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
    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed", type=int, default=3023)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-rslora", action="store_true")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    # parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, required=True)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

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

    training_arguments = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        # num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to="none", # Use this for WandB etc
        save_total_limit=3,
        save_steps=500,
    )

    model, tokenizer = run_finetuning(args.model_name, lora_arguments, training_arguments, args.seed,
                                      args.max_seq_length, dtype, args.load_in_4bit, args.device, args.output_dir)

    sample_inference(model, tokenizer)