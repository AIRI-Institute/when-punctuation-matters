mkdir train_test_splits

for n_formats in 10 50
do
    for mode in random compositional_separator_space compositional_separator unseen_space unbalanced_random
    do
        output_dir=train_test_splits

        mkdir ${output_dir}/${mode}

        python generate_test_formats.py \
            --task_filename DUMMY_VALUE \
            --dataset_name natural-instructions \
            --num_formats_to_analyze ${n_formats} \
            --cache_dir /home/mvchaychuk/.cache/huggingface \
            --output_dir ${output_dir} \
            --n-train 0 \
            --n-val 0 \
            --n-test ${n_formats} \
            --seed 23 \
            --mode ${mode}
    done
done
