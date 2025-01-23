import os
import json
import copy
import random
import itertools
from typing import List, Dict

from main import (
    _load_task,
    pointers_to_all_objects,
    create_pointer_action_type_pairs,
    holistic_node_format_sanity_checks,
    make_parser,
    _get_task_filename_to_print
)
from grammar_definition import NewEnumerationPromptFormat

### Constants

CHOSEN_SEPARATOR_LIST = ['', '::: ', ':: ', ': ', ' \n\t', '\n    ', ' : ', ' - ' , ' ', '\n ', '\n\t', ':', '::', '- ', '\t']  # sep='' is used rarely, only for enumerations because there is already formatting there
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

### Short - Long + unseen '\n' generalization

SHORT_TRAIN_SEPARATOR_LIST = ['', ': ', ' - ', ' ', ':', '- ',  '::: ', ':: ', '::']
LONG_TEST_SEPARATOR_LIST = [' \n\t', '\n    ', ' : ', '\n\t', '\t', '\n ']

SHORT_TRAIN_SPACE_LIST = ['', ' ', ' -- ', ' || ', ' <sep> ', ', ', ' , ', '. ']
LONG_TEST_SPACE_LIST = [' \n', '  ', '; \n', ' \n ', '\n ', '\n', ' ,  ']

### Compositional generalization

COMPOSITIONAL_TRAIN_SEPARATOR_LIST = ['', ' ', ':', '- ', '\t', '\n ']
COMPOSITIONAL_TEST_SEPARATOR_LIST = [' - ', ' \n\t', '\n    ', ' : ', '\n\t',  '::: ', ':: ', ': ', '::']

COMPOSITIONAL_TRAIN_SPACE_LIST = ['', ' ', ' -- ', ' || ', ' <sep> ', '\n', ', ', '. ', ';']
COMPOSITIONAL_TEST_SPACE_LIST = [' \n', '  ', '; \n', ' \n ', '\n ', ' , ', ' ,  ']

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
    else:
        valid_value_assignments = set()
        while len(valid_value_assignments) < args.num_formats_to_analyze:
            value_assignment = [random.choice(sublist) for sublist in action_value_options]
            if _value_assignment_is_valid(
                    structured_prompt_format, global_constraints, value_assignment, args.allow_text_action_type, mapping_all_categories):
                valid_value_assignments.add(tuple(value_assignment))
        valid_value_assignments = [list(e) for e in valid_value_assignments]

    # set an order in which to shuffle the whole dataset (including demonstrations)
    dataset_ordered_ids = list(range(raw_dataset_size))
    random.shuffle(dataset_ordered_ids)

    return valid_value_assignments, dataset_ordered_ids, pointer_action_pairs


def _value_assignment_is_valid(structured_prompt_format, global_constraints, value_assignment, allow_text_action_type, mapping_all_categories):
    # A. copy structured_prompt_format to avoid modifying the original
    new_structured_prompt_format, new_global_constraints = \
        copy.deepcopy((structured_prompt_format, global_constraints))
    all_pointers = pointers_to_all_objects(new_structured_prompt_format) + new_global_constraints
    all_pointers_enumerated = [(e, i) for i, e in enumerate(all_pointers)]
    pointer_action_pairs = create_pointer_action_type_pairs(
        all_pointers_enumerated, allow_text_action_type=allow_text_action_type)

    # B. apply the value assignment
    value_assignments_ids = value_assignment_str_to_indices([value_assignment], pointer_action_pairs, mapping_all_categories)[0]
    for (element, element_id, action_type), action_value_id in zip(pointer_action_pairs, value_assignments_ids):
        action_value, action_value_name = mapping_all_categories[action_type][int(action_value_id)]
        element.update_field(action_type, action_value)

    # C. evaluate new node holistically
    return holistic_node_format_sanity_checks(new_structured_prompt_format)


def value_assignment_str_to_indices(value_assignments, pointer_action_pairs, mapping_all_categories):
    value_assignments_ids = []
    for assignment in value_assignments:
        assert len(pointer_action_pairs) == len(assignment), f"{len(pointer_action_pairs)} != {len(assignment)}"
        assignment_ids = []
        for (_, _, action_type), assignment_value in zip(pointer_action_pairs, assignment):
            idx = [i for i, (_, v) in enumerate(mapping_all_categories[action_type]) if v == assignment_value][0]
            assignment_ids.append(idx)
        value_assignments_ids.append(assignment_ids)
    return value_assignments_ids


def save_json(data, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def process_task(args, reference_action_type2test_elements: Dict[str, List[str]] | None) -> Dict:
    """ Generates and saves a json file with test prompt formats, dataset examples order and some metadata (action_types).
    Inputs:
        args:
            Mostly defined in main.py, with several arguments added below
        reference_action_type2test_elements:
            A mapping from action type (e.g. 'chose_space') to a list of possible options (e.g. [' ', '\n', '\t']).
            Used to ensure consistent test formats across all tasks.
    Outputs:
        A dict with train, validation, test prompt formats, dataset examples order and some metadata (action_types).
    """
    task_filename_to_print = _get_task_filename_to_print(args)
    disable_text_action_type = 'textdisabled'
    metadata_filename = "metadata.txt"

    random.seed(args.seed)

    print(f"Processing task {args.task_filename}")

    filepath = os.path.join(args.output_dir, 
                            f'holistic_random_sample_{task_filename_to_print}_nodes_{args.num_formats_to_analyze}_{disable_text_action_type}.json')

    if args.mode == "random":
        args.num_formats_to_analyze = args.n_train + args.n_val + args.n_test
        value_assignments, dataset_ordered_ids, pointer_action_pairs = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES)
        test = value_assignments[args.n_train + args.n_val:]

        # If we have reference values for test formats, patch test
        if reference_action_type2test_elements:
            action_types = [action_type for _, _, action_type in pointer_action_pairs]

            test = []
            for index in range(args.n_test):
                format = [reference_action_type2test_elements[action][index] for action in action_types]
                test.append(format)

        # Train will be defined implicitely, as a complement to test
        train = []
    elif args.mode == "space":
        args.num_formats_to_analyze = args.n_train
        train, dataset_ordered_ids, pointer_action_pairs = _sample_value_assignments(args, SHORT_TRAIN_MAPPING_space)
        args.num_formats_to_analyze = args.n_test
        test, _, _ = _sample_value_assignments(args, LONG_TEST_MAPPING_space)

    elif args.mode == "separator_space":
        args.num_formats_to_analyze = args.n_train
        train, dataset_ordered_ids, pointer_action_pairs = _sample_value_assignments(args, SHORT_TRAIN_MAPPING_space_separator)
        args.num_formats_to_analyze = args.n_test
        test, _, _ = _sample_value_assignments(args, LONG_TEST_MAPPING_space_separator)

    elif args.mode == "compositional_separator":
        train_seps = [(e, e) for e in COMPOSITIONAL_TRAIN_SEPARATOR_LIST]
        test_seps = [(e, e) for e in COMPOSITIONAL_TEST_SEPARATOR_LIST]

        args.num_formats_to_analyze = args.n_train
        train, dataset_ordered_ids, pointer_action_pairs = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES | {"chosen_separator": train_seps})

        args.num_formats = args.n_test
        test, _, _ = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES | {"chosen_separator": test_seps})

        with open(f"{args.output_dir}/{metadata_filename}", "w") as f:
            print("Train separators:", [e[0] for e in train_seps], file=f)
            print("Test separators:", [e[0] for e in test_seps], file=f)

    elif args.mode == "compositional_separator_space":
        train_seps = [(e, e) for e in COMPOSITIONAL_TRAIN_SEPARATOR_LIST]
        test_seps = [(e, e) for e in COMPOSITIONAL_TEST_SEPARATOR_LIST]
        train_spaces = [(e, e) for e in COMPOSITIONAL_TRAIN_SPACE_LIST]
        test_spaces = [(e, e) for e in COMPOSITIONAL_TEST_SPACE_LIST]

        args.num_formats_to_analyze = args.n_train
        train, dataset_ordered_ids, pointer_action_pairs = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES | {"chosen_separator": train_seps, "chosen_space": train_spaces})

        args.num_formats = args.n_test
        test, _, _ = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES | {"chosen_separator": test_seps, "chosen_space": test_spaces})

        with open(f"{args.output_dir}/{metadata_filename}", "w") as f:
            print("Train separators:", [e[0] for e in train_seps], file=f)
            print("Test separators:", [e[0] for e in test_seps], file=f)
            print("Train spaces:", [e[0] for e in train_spaces], file=f)
            print("Test spaces:", [e[0] for e in test_spaces], file=f)
    elif args.mode == "unseen_space":
        train_spaces = [("\n", "\n")]
        test_spaces = [e for e in CHOSEN_SPACE_LIST if e[0] != "\n"]

        args.num_formats_to_analyze = args.n_train
        train, dataset_ordered_ids, pointer_action_pairs = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES | {"chosen_space": train_spaces})

        args.num_formats = args.n_test
        test, _, _ = _sample_value_assignments(args, VANILLA_MAPPING_ALL_CATEGORIES | {"chosen_space": test_spaces})

        with open(f"{args.output_dir}/{metadata_filename}", "w") as f:
            print("Train spaces:", [e[0] for e in train_spaces], file=f)
            print("Test spaces:", [e[0] for e in test_spaces], file=f)
    else:
        raise ValueError(f"{args.mode=} is not supported yet.")

    data = {
        "train_formats": train,
        "val_formats": [],
        "test_formats": test,
        "action_types": [action_type for _, _, action_type in pointer_action_pairs],
        "dataset_ordered_ids": dataset_ordered_ids,
    }

    save_json(data, filepath)

    return data


def build_reference_action_type2test_elements(task070_data, mode):
    if mode != "random":
        return None

    reference_action_type2test_elements = {}
    for action_type in task070_data["action_types"]:
        reference_action_type2test_elements[action_type] = []

    for format in task070_data["test_formats"]:
        for element, action_type in zip(format, task070_data["action_types"]):
            reference_action_type2test_elements[action_type].append(element)

    """
    Task 70 has following action types:
       ['chosen_space',
        'chosen_item_wrapper',
        'chosen_number_format',
        'chosen_space',
        'chosen_separator',
        'chosen_separator_text_and_option',
        'text_descriptor_fn',
        'chosen_separator']
    'chosen_space' and 'chosen_separator' are mentioned twice, and the code above appends elements 
    from both instances of e.g. 'chosen_space' in one list, resulting in more than 'args.n_test' options.
    So we need to truncate them.
    """
    reference_action_type2test_elements = {
        key: value[:args.n_test] for key, value in reference_action_type2test_elements.items()
    }

    for element_options in reference_action_type2test_elements.values():
        print(len(element_options))
    assert all(len(element_options) == args.n_test for element_options in reference_action_type2test_elements.values())

    return reference_action_type2test_elements

# "task190" is excluded due to incorrect labels
TASK_NAMES = ["task050", "task065", "task069", "task070", "task114", "task133", "task155", "task158", "task161", "task162", "task163", "task213", "task214", "task220", "task279", "task280", "task286", "task296", "task297", "task316", "task317", "task319", "task320", "task322", "task323", "task325", "task326", "task327", "task328", "task335", "task337", "task385", "task580", "task607", "task608", "task609", "task904", "task905", "task1186", "task1283", "task1284", "task1297", "task1347", "task1387", "task1419", "task1420", "task1421", "task1423", "task1502", "task1612", "task1678", "task1724"]

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
    original_n_nodes = args.num_formats_to_analyze

    # For "random" mode, we need to unify the test formats across all tasks.
    # So we sample concrete formats for task070 (which has all possible prompt components)
    # and then use the same formats for other tasks (dropping non-applicable elements if needed,
    # e.g. discarding components related to option formatting).
    # NOTE: due to implementation details, test formats obtained for task 070 with `reference_action_type2test_elements`=None
    # are different from test formats for task 070 with non-None `reference_action_type2test_elements`.
    args.task_filename = "task070_"
    task070_data = process_task(args, reference_action_type2test_elements=None)
    reference_action_type2test_elements = build_reference_action_type2test_elements(task070_data, args.mode)

    # Generate formats
    for task_name in TASK_NAMES:
        args.task_filename = task_name + "_"
        args.num_formats_to_analyze = original_n_nodes
        process_task(args, reference_action_type2test_elements)
