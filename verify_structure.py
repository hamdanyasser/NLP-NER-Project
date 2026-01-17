"""
Verification script to check project structure and imports.
This runs basic checks without requiring full dependencies.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_structure():
    """Verify project directory structure."""
    print("\n" + "="*60)
    print("Checking Directory Structure")
    print("="*60)

    dirs = [
        "config",
        "data/raw",
        "data/processed",
        "src/data",
        "src/models",
        "src/training",
        "src/utils",
        "notebooks",
        "reports",
        "scripts",
        "artifacts"
    ]

    all_good = True
    for dir_path in dirs:
        exists = os.path.isdir(dir_path)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}/")
        all_good = all_good and exists

    return all_good

def check_core_files():
    """Verify core Python files exist."""
    print("\n" + "="*60)
    print("Checking Core Files")
    print("="*60)

    files = [
        ("README.md", "README"),
        ("requirements.txt", "Requirements"),
        ("config/config.yaml", "Configuration"),
        ("src/data/preprocess.py", "Data preprocessing"),
        ("src/data/dataset.py", "Dataset module"),
        ("src/utils/vocab.py", "Vocabulary utilities"),
        ("src/utils/metrics.py", "Metrics module"),
        ("src/utils/logging_utils.py", "Logging utilities"),
        ("src/models/layers.py", "CRF layer"),
        ("src/models/baseline_tagger.py", "Baseline model"),
        ("src/models/bilstm_crf.py", "BiLSTM-CRF model"),
        ("src/training/train.py", "Training script"),
        ("src/training/eval.py", "Evaluation script"),
        ("scripts/run_train.sh", "Training shell script"),
        ("scripts/run_eval.sh", "Evaluation shell script"),
        ("reports/report_outline.md", "Report outline"),
        ("reports/experiments.md", "Experiments log"),
        ("notebooks/exploration.ipynb", "Exploration notebook"),
    ]

    all_good = True
    for filepath, description in files:
        exists = check_file_exists(filepath, description)
        all_good = all_good and exists

    return all_good

def check_data_files():
    """Check if processed data files exist."""
    print("\n" + "="*60)
    print("Checking Data Files")
    print("="*60)

    data_files = [
        ("data/processed/train.txt", "Training data"),
        ("data/processed/dev.txt", "Development data"),
        ("data/processed/test.txt", "Test data"),
    ]

    all_good = True
    for filepath, description in data_files:
        exists = check_file_exists(filepath, description)
        all_good = all_good and exists

        # If file exists, show first few lines
        if exists:
            with open(filepath, 'r') as f:
                lines = [f.readline().strip() for _ in range(3)]
            print(f"  Preview: {lines[0]}")

    return all_good

def check_python_syntax():
    """Check Python files for syntax errors."""
    print("\n" + "="*60)
    print("Checking Python Syntax")
    print("="*60)

    python_files = [
        "src/data/preprocess.py",
        "src/data/dataset.py",
        "src/utils/vocab.py",
        "src/utils/metrics.py",
        "src/utils/logging_utils.py",
        "src/models/layers.py",
        "src/models/baseline_tagger.py",
        "src/models/bilstm_crf.py",
        "src/training/train.py",
        "src/training/eval.py",
    ]

    all_good = True
    for filepath in python_files:
        try:
            with open(filepath, 'r') as f:
                compile(f.read(), filepath, 'exec')
            print(f"✓ {filepath}")
        except SyntaxError as e:
            print(f"✗ {filepath}: {e}")
            all_good = False

    return all_good

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("BiLSTM-CRF NER Project Structure Verification")
    print("="*60)

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Core Files", check_core_files),
        ("Data Files", check_data_files),
        ("Python Syntax", check_python_syntax),
    ]

    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    all_passed = all(results.values())
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check_name}: {status}")

    print("="*60)

    if all_passed:
        print("\n✓ All checks passed! Project structure is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Preprocess data: python -m src.data.preprocess")
        print("3. Train baseline: python -m src.training.train --model baseline")
        print("4. Train BiLSTM-CRF: python -m src.training.train --model bilstm_crf")
        print("5. Evaluate: python -m src.training.eval")
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
