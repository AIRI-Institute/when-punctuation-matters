import os
import json
import random
import itertools

from main import (
    _load_task,
    pointers_to_all_objects,
    create_pointer_action_type_pairs,
    _value_assignment_is_valid,
    make_parser,
    _get_task_filename_to_print
)
from grammar_definition import NewEnumerationPromptFormat

### Constants

CHOSEN_SEPARATOR_LIST = ['', '::: ', ':: ', ': ', ' \n\t', '\n    ', ' : ', ' - ', ' ', '\n ', '\n\t', ':', '::', '- ', '\t']  # sep='' is used rarely, only for enumerations because there is already formatting there
CHOSEN_SPACE_LIST = ['', ' ', '\n', ' \n', ' -- ',  '  ', '; \n', ' || ', ' <sep> ', ' -- ', ', ', ' \n ', ' , ', '\n ', '. ', ' ,  ']  # space='' is used a lot
CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST = ['', ' ', '  ', '\t']

CHOSEN_SEPARATOR_LIST = [(e, e) for e in CHOSEN_SEPARATOR_LIST]
CHOSEN_SPACE_LIST = [(e, e) for e in CHOSEN_SPACE_LIST]
CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST = [(e, e) for e in CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST]


TEXT_DESCRIPTOR_FN_LIST = [
    (lambda x: x, "lambda x: x"),
    (lambda x: x.title(), "lambda x: x.title()"),
    (lambda x: x.upper(), "lambda x: x.upper()"),
    (lambda x: x.lower(), "lambda x: x.lower()")
]
ITEM_WRAPPER_LIST = [
    (lambda x: f'({x})', "lambda x: f'({x})'"),
    (lambda x: f'{x}.', "lambda x: f'{x}.'"),
    (lambda x: f'{x})', "lambda x: f'{x})'"),
    (lambda x: f'{x} )', "lambda x: f'{x} )'"),
    (lambda x: f'[{x}]', "lambda x: f'[{x}]'"),
    (lambda x: f'<{x}>', "lambda x: f'<{x}>'"),
]
NUMBER_FORMAT_LIST = [
    (lambda x: x + 1, "lambda x: x + 1"),
    (lambda x: chr(ord('A') + x), "lambda x: chr(ord('A') + x)"),
    (lambda x: chr(ord('a') + x), "lambda x: chr(ord('a') + x)"),
    (lambda x: chr(0x215F + x + 1) + ('' if x < 12 else 0 / 0), "lambda x: chr(0x215F + x + 1)"),
    (lambda x: NewEnumerationPromptFormat.ROMAN_NUMERALS[x], "lambda x: EnumerationPromptFormat.ROMAN_NUMERALS[x]"),
    (lambda x: NewEnumerationPromptFormat.ROMAN_NUMERALS[x].upper(), "lambda x: EnumerationPromptFormat.ROMAN_NUMERALS[x].upper()")
]

### Short - Long generalization

SHORT_TRAIN_SEPARATOR_LIST = ['', ': ', ' - ', ' ', ':', '- ',  '::: ', ':: ', '::']
LONG_TEST_SEPARATOR_LIST = [' \n\t', '\n    ', ' : ', '\n\t', '\t', '\n ']

SHORT_TRAIN_SPACE_LIST = ['', ' ', ' -- ', ' || ', ' <sep> ', ' -- ', ', ', ' , ', '. ']
LONG_TEST_SPACE_LIST = [' \n', '  ', '; \n', ' \n ', '\n ', '\n', ' ,  ']

### Mappings

VANILLA_MAPPING_ALL_CATEGORIES = {
    'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST,
    'chosen_item_wrapper': ITEM_WRAPPER_LIST,
    'chosen_number_format': NUMBER_FORMAT_LIST,
    'chosen_space': CHOSEN_SPACE_LIST,
    'chosen_separator': CHOSEN_SEPARATOR_LIST,  # in OPTION_1:^TEXT, this is ^
    'chosen_separator_text_and_option': CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST  # in OPTION_1:^TEXT, this is _
}

SHORT_TRAIN_MAPPING_space = {
    'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST,
    'chosen_item_wrapper': ITEM_WRAPPER_LIST,
    'chosen_number_format': NUMBER_FORMAT_LIST,
    'chosen_space': [(e, e) for e in SHORT_TRAIN_SPACE_LIST],
    'chosen_separator': CHOSEN_SEPARATOR_LIST,  # in OPTION_1:^TEXT, this is ^
    'chosen_separator_text_and_option': CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST  # in OPTION_1:^TEXT, this is _
}
LONG_TEST_MAPPING_space = {
    'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST,
    'chosen_item_wrapper': ITEM_WRAPPER_LIST,
    'chosen_number_format': NUMBER_FORMAT_LIST,
    'chosen_space': [(e, e) for e in LONG_TEST_SPACE_LIST],
    'chosen_separator': CHOSEN_SEPARATOR_LIST,  # in OPTION_1:^TEXT, this is ^
    'chosen_separator_text_and_option': CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST  # in OPTION_1:^TEXT, this is _
}

SHORT_TRAIN_MAPPING_space_separator = {
    'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST,
    'chosen_item_wrapper': ITEM_WRAPPER_LIST,
    'chosen_number_format': NUMBER_FORMAT_LIST,
    'chosen_space': [(e, e) for e in SHORT_TRAIN_SPACE_LIST],
    'chosen_separator': [(e, e) for e in SHORT_TRAIN_SEPARATOR_LIST],  # in OPTION_1:^TEXT, this is ^
    'chosen_separator_text_and_option': CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST  # in OPTION_1:^TEXT, this is _
}
LONG_TEST_MAPPING_space_separator = {
    'text_descriptor_fn': TEXT_DESCRIPTOR_FN_LIST,
    'chosen_item_wrapper': ITEM_WRAPPER_LIST,
    'chosen_number_format': NUMBER_FORMAT_LIST,
    'chosen_space': [(e, e) for e in LONG_TEST_SPACE_LIST],
    'chosen_separator': [(e, e) for e in LONG_TEST_SEPARATOR_LIST],  # in OPTION_1:^TEXT, this is ^
    'chosen_separator_text_and_option': CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST  # in OPTION_1:^TEXT, this is _
}

### Code

def _sample_value_assignments(args, mapping_all_categories):
    # load task [we might do it twice, but this first time is to load the structured_prompt_format]
    structured_prompt_format, global_constraints, extra_params_structured_prompt_format, \
        original_multiple_choice_output_format, args_compute_node_score, raw_dataset_size = _load_task(args)

    # sample nodes to evaluate if file has not been passed
    all_pointers = pointers_to_all_objects(structured_prompt_format) + global_constraints
    all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]
    pointer_action_pairs = create_pointer_action_type_pairs(
        all_pointers_enumerated, allow_text_action_type=args.allow_text_action_type)

    action_value_options = []
    for a, b, action_type in pointer_action_pairs:
        action_value_options.append([f_name for f_value, f_name in mapping_all_categories[action_type]])

    num_combinations = 1
    for e in action_value_options:
        num_combinations *= len(e)

    if num_combinations <= args.num_formats_to_analyze:
        raise ValueError(f"Not enough prompt formats: {num_combinations=}, required {args.num_formats_to_analyze}")
        valid_value_assignments = []
        for value_assignment in itertools.product(*action_value_options):
            if _value_assignment_is_valid(
                    structured_prompt_format, global_constraints, value_assignment, args.allow_text_action_type):
                valid_value_assignments.append(value_assignment)
    else:
        valid_value_assignments = set()
        while len(valid_value_assignments) < args.num_formats_to_analyze:
            value_assignment = [random.choice(sublist) for sublist in action_value_options]
            if _value_assignment_is_valid(
                    structured_prompt_format, global_constraints, value_assignment, args.allow_text_action_type):
                valid_value_assignments.add(tuple(value_assignment))
        valid_value_assignments = [list(e) for e in valid_value_assignments]

    # set an order in which to shuffle the whole dataset (including demonstrations)
    dataset_ordered_ids = list(range(raw_dataset_size))
    random.shuffle(dataset_ordered_ids)

    return valid_value_assignments, dataset_ordered_ids


def save_json(data, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-val", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--mode")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    args.disable_text_action_type = True
    args.allow_text_action_type = not args.disable_text_action_type
    disable_text_action_type = 'textdisabled'
    task_filename_to_print = _get_task_filename_to_print(args)

    random.seed(args.seed)

    print(f"Processing task {args.task_filename}")

    filepath = os.path.join(args.output_dir, 
                            f'holistic_random_sample_{task_filename_to_print}_nodes_{args.num_formats_to_analyze}_{disable_text_action_type}.json')

    # random split
    if args.mode == "random":
        args.num_formats_to_analyze = args.n_train + args.n_val + args.n_test
        valid_value_assignments, dataset_ordered_ids = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES)

        train = valid_value_assignments[:args.n_train]
        val = valid_value_assignments[args.n_train:args.n_train + args.n_val]
        test = valid_value_assignments[args.n_train + args.n_val:args.n_train + args.n_val + args.n_test]
    elif args.mode == "space":
        args.num_formats_to_analyze = args.n_train
        train, dataset_ordered_ids = _sample_value_assignments(args, SHORT_TRAIN_MAPPING_space)
        args.num_formats_to_analyze = args.n_test
        test, _ = _sample_value_assignments(args, LONG_TEST_MAPPING_space)

        val = []
    elif args.mode == "space_separator":
        args.num_formats_to_analyze = args.n_train
        train, dataset_ordered_ids = _sample_value_assignments(args, SHORT_TRAIN_MAPPING_space_separator)
        args.num_formats_to_analyze = args.n_test
        test, _ = _sample_value_assignments(args, LONG_TEST_MAPPING_space_separator)

        val = []
    else:
        raise ValueError(f"{args.mode=} is not supported yet.")

    data = {
        "train_formats": train,
        "val_formats": val,
        "test_formats": test,
        "dataset_ordered_ids": dataset_ordered_ids,
    }

    save_json(data, filepath)
