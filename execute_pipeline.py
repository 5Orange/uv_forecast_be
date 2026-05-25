import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"ERROR {description} failed")
        return False

    return True


def main():
    steps = [
        ("python -m src.pipeline", "Feature regeneration"),
        # ("python -m src.models.run_tuning", "Hyperparameter Tuning"),
        ("python -m src.models.train_pipeline", "Model training and evaluation"),
    ]

    for cmd, desc in steps:
        if not run_command(cmd, desc):
            sys.exit(1)

    print("\nResults: results/model_comparison.csv")

if __name__ == "__main__":
    main()
