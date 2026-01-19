#!/usr/bin/env python3
import os
import sys
import subprocess

def print_help():
    print("Usage: tremm <subcommand> [options]")
    print("\nAvailable Subcommands:")
    print("  list                      List available datasets.")
    print("  create_dataset            Create a dataset.")
    print("  create_kfold              Create a k-fold cross-validation dataset.")
    print("  create_n_kfold            Create n_kfold dataset with global validation set.")
    print("Examples:")
    print("  tremm list")
    print("  tremm create_dataset --dataset_size 100000 --split_ratios 0.7 0.15 0.15 --positive_ratios 0.5 0.25 0.25 --random_seed 42")
    print("  tremm create_kfold --dataset_size 100000 --ext_test_ratio 0.2 --int_valid_ratio 0.2 --random_seed 42")
    print("  tremm create_n_kfold --dataset_size 100000 --ext_test_ratio 0.2 --global_valid_ratio 0.1 --num_internal_folds 5 --int_valid_ratio 0.2 --random_seed 42")
    sys.exit(0)

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print_help()

    SUBCOMMAND_MODULE_MAP = {
        "list": "tremm.scripts.dataset_manager",
        "create_dataset": "tremm.scripts.dataset_manager",
        "create_kfold": "tremm.scripts.dataset_manager",
        "create_n_kfold": "tremm.scripts.dataset_manager",
    }

    subcommand = sys.argv[1]

    if subcommand not in SUBCOMMAND_MODULE_MAP:
        print(f"Unknown subcommand: {subcommand}\n")
        print_help()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    os.chdir(project_root)

    python_module = SUBCOMMAND_MODULE_MAP[subcommand]

    command = ["python3", "-m", python_module, subcommand] + sys.argv[2:]

    result = subprocess.run(command, check=False)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
