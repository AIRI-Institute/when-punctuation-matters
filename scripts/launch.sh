devices=$1                      # e.g. "0"
full_huggingface_model_name=$2  # e.g. "unsloth/Llama-3.2-1B-Instruct"
n_shot=$3                       # e.g. "2"
format_split_mode=$4            # e.g. "random"
suffix=$5                       # e.g. "---no-chat-template"
apply_batch_calibration=$6      # "1" to turn on, "0" to turn off

# Splits by `/` and takes last part (which is model's name)
exp_name=$( echo $full_huggingface_model_name | rev | cut -d / -f1 | rev )
exp_name="${exp_name}${suffix}"

num_formats_to_analyze=9

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LRU_CACHE_CAPACITY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DISABLE_CHAT_TEMPLATE="1"

echo "which python:" $(which python)
echo "parsed experiment name:" $exp_name

if [[ ${DISABLE_CHAT_TEMPLATE} == "1" ]]
then
    echo "DISABLE_CHAT_TEMPLATE = 1 => FORCED NO CHAT TEMPLATE FOR ALL MODELS"
fi

# A hack to speed up experiments: everything is first launched with large batch, some runs fall with OOM, but get re-launched in next iteration.
# Results for finished tasks are not re-computed.
for batch_size_llm in 64 32 16 8
do
    for task in task050 task065 task069 task070 task114 task133 task155 task158 task161 task162 task163 task190 task213 task214 task220 task279 task280 task286 task296 task297 task316 task317 task319 task320 task322 task323 task325 task326 task327 task328 task335 task337 task385 task580 task607 task608 task609 task904 task905 task1186 task1283 task1284 task1297 task1347 task1387 task1419 task1420 task1421 task1423 task1502 task1612 task1678 task1724
    do
        CUDA_VISIBLE_DEVICES=$devices python main.py \
            --task_filename ${task}_ \
            --dataset_name natural-instructions \
            --num_formats_to_analyze ${num_formats_to_analyze} \
            --batch_size_llm ${batch_size_llm} \
            --num_samples 1000 \
            --model_name ${full_huggingface_model_name} \
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