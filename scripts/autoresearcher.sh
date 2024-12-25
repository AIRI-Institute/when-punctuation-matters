### Input parameters ###
devices=$1                      # e.g. "0"
full_huggingface_model_name=$2  # e.g. "unsloth/Llama-3.2-3B-Instruct"
format_split_mode=$3            # e.g. "random"
finetune_output_dir=$4          # e.g. "llama1b_iid"
suffix=$5                       # e.g. "---iid-no-chat-template"

### Setting some variables ###
dataset="data/df_hermes_simple_answers.csv"
# Splits by `/` and takes last part (which is model's name)
model_name=$( echo ${full_huggingface_model_name} | rev | cut -d / -f1 | rev )
finetuned_model_name=${model_name}_lora
exp_name=${finetuned_model_name}${suffix}

num_formats_to_analyze=9
n_shot=0
apply_batch_calibration=0

if [[ ${format_split_mode} == "random" ]]; then
    path_to_test_formats=train_test_splits/random/holistic_random_sample_task050_nodes_${num_formats_to_analyze}_textdisabled.json
else
    path_to_test_formats=""
fi

### Setting environment variables ###
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LRU_CACHE_CAPACITY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DISABLE_CHAT_TEMPLATE="1"

if [[ ${DISABLE_CHAT_TEMPLATE} == "1" ]]
then
    echo "DISABLE_CHAT_TEMPLATE = 1 => FORCED NO CHAT TEMPLATE FOR ALL MODELS"
fi

### Coherence check prints ###
echo "which python:" $(which python)
echo "parsed experiment name:" $exp_name

### Training ###

expected_checkpoint_location=training/${finetune_output_dir}/${finetuned_model_name}

for retry in $(seq 1 48)
do
    # If finetune was successful, there will be a directory for the checkpoint:
    if [ -d ${expected_checkpoint_location} ]; then
        echo "================================"
        echo "Finished finetuning"
        echo "================================"
        break
    fi

    CUDA_VISIBLE_DEVICES=$devices python finetune.py \
        --model-name ${full_huggingface_model_name} \
        --dataset-name ${dataset} \
        --output-dir training/${finetune_output_dir} \
        --format-split-mode ${format_split_mode} \
        --path-to-test-formats ${path_to_test_formats}

    # Otherwise, retry finetuning after 30 minute pause
    echo "Failed retry ${retry}"
    sleep 30m
done


### Evaluation ###

# A hack to speed up experiments: everything is first launched with large batch, some runs fall with OOM, but get re-launched in next iteration.
# Results for finished tasks are not re-computed.
for batch_size_llm in 64 32 16 8
do
    for task in task050 task065 task069 task070 task114 task133 task155 task158 task161 task162 task163 task190 task213 task214 task220 task279 task280 task286 task296 task297 task316 task317 task319 task320 task322 task323 task325 task326 task327 task328 task335 task337 task385 task580 task607 task608 task609 task904 task905 task1186 task1283 task1284 task1297 task1347 task1387 task1419 task1420 task1421 task1423 task1502 task1612 task1678 task1724
    do
        CUDA_VISIBLE_DEVICES=$devices python main.py \
            --model_name ${expected_checkpoint_location} \
            --task_filename ${task}_ \
            --dataset_name natural-instructions \
            --num_formats_to_analyze ${num_formats_to_analyze} \
            --batch_size_llm ${batch_size_llm} \
            --num_samples 1000 \
            --n_shot ${n_shot} \
            --num_ensembles 4 \
            --ensemble_size 5 \
            --apply_batch_calibration ${apply_batch_calibration} \
            --evaluation_metric probability_ranking \
            --evaluation_type full \
            --cache_dir /home/seleznev/.cache/huggingface \
            --output_dir exp/${exp_name} \
            --nodes_to_evaluate_filepath train_test_splits/${format_split_mode}/holistic_random_sample_${task}_nodes_${num_formats_to_analyze}_textdisabled.json
    done
done