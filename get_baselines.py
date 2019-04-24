import subprocess

from classifiers import CLASSIFIERS
from datasets import DATASETS


for dataset in DATASETS:
    print(dataset[0])
    for classifier in CLASSIFIERS.keys():
        try:
            subprocess.run(['python', 'run_baseline.py', dataset[0], dataset[1], classifier], timeout=1800)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
