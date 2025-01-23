mkdir train_test_splits

for mode in random compositional_separator_space compositional_separator unseen_space
do
    output_dir=train_test_splits/${mode}
    mkdir ${output_dir}

    python generate_train_val_test_formats.py \
        --task_filename DUMMY_VALUE \
        --dataset_name natural-instructions \
        --num_formats_to_analyze 10 \
        --cache_dir /home/seleznev/.cache/huggingface \
        --output_dir ${output_dir} \
        --n-train 0 \
        --n-val 0 \
        --n-test 10 \
        --seed 23 \
        --mode ${mode}
done