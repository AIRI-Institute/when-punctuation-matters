import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from datasets import load_dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq

def load_model(name, max_seq_length, dtype, load_in_4bit, device):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model.to(device)

    return model, tokenizer


def get_hermes_dataset(tokenizer):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    def formatting_prompts_func(examples):
        conversations = examples["conversations"]
        translate = {"human": "user", "gpt": "assistant"}
        reformatted_conversations = []
        for conversation in conversations:
            # TODO: insert augmentations here
            # - fix random seed
            # - how to handle multiple-choice queries?
            reformatted_conversations.append(
                [
                    {
                        "role": msg["from"] if msg["from"] not in translate else translate[msg["from"]],
                        "content": msg["value"]
                    }
                for msg in conversation]
            )
            
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in reformatted_conversations]
        return { "text" : texts, }

    dataset_name = "teknium/OpenHermes-2.5"
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

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


def run_finetuning(name: str, lora_arguments: LoraArguments, training_arguments: TrainingArguments,
                   seed: int, max_seq_length: int, dtype: torch.dtype, load_in_4bit: bool, device: str):
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

    dataset = get_hermes_dataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc=2,
        packing=False, # Can make training 5x faster for short sequences.
        args=training_arguments
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    print(f"{tokenizer.decode(trainer.train_dataset[5]['input_ids'])=}")
    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    print(f"{tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]['labels']])=}")

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
    model.save_pretrained(f"checkpoints/{model_name}_lora")
    tokenizer.save_pretrained(f"checkpoints/{model_name}_lora")
    # model.push_to_hub("your_name/lora_model", token = "...")
    # tokenizer.push_to_hub("your_name/lora_model", token = "...")


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
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.2",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 1.5, min_p = 0.1)
    generation = tokenizer.batch_decode(outputs)

    print(f"{generation=}")


if __name__ == "__main__":
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    seed = 3023
    max_seq_length = 2048
    dtype = torch.bfloat16
    load_in_4bit = False
    device = "cuda:0"

    lora_arguments = LoraArguments(
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        lora_rank=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
    )

    training_arguments = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs=1,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        output_dir="training_outputs",
        report_to="none", # Use this for WandB etc
    )

    model, tokenizer = run_finetuning(model_name, lora_arguments, training_arguments,
                                      seed, max_seq_length, dtype, load_in_4bit, device)

    sample_inference(model, tokenizer)