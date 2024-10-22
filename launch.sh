devices=$1

for task in task050 task065 task069 task070 task114 task133 task155 task158 task161 task162 task163 task190 task213 task214 task220 task279 task280 task286 task296 task297 task316 task317 task319 task320 task322 task323 task325 task326 task327 task328 task335 task337 task385 task580 task607 task608 task609 task904 task905 task1186 task1283 task1284 task1297 task1347 task1387 task1419 task1420 task1421 task1423 task1502 task1612 task1678 task1724
do
    CUDA_VISIBLE_DEVICES=$devices python main.py \
        --task_filename ${task}_ \
        --dataset_name natural-instructions \
        --num_formats_to_analyze 9 \
        --batch_size_llm 2 \
        --num_samples 1000 \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --n_shot 5 \
        --evaluation_metric probability_ranking \
        --evaluation_type full \
        --cache_dir /home/seleznev/.cache/huggingface \
        --output_dir data/qwen2.5-7b
done