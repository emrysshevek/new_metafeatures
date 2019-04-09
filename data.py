import os
import json
import signal
import time

import pandas as pd
import numpy as np
from metalearn import Metafeatures

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

import warnings
from sklearn.exceptions import DataConversionWarning, FitFailedWarning, ConvergenceWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASETS_DIR = '/users/data/d3m/datasets/training_datasets/LL0'
TIMEOUT = 'TIMEOUT'
ERROR = 'ERROR'

def timeout_handler(signum, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, timeout_handler)


def data_iterator():
    for name in os.listdir(DATASETS_DIR):
        data_path = os.path.join(DATASETS_DIR, name)
        if os.path.isdir(data_path):
            csv_path = os.path.join(data_path, name + '_dataset/tables/learningData.csv')
            info_path = os.path.join(data_path, name + '_problem/problemDoc.json')
            with open(info_path, 'r') as fp:
                info_file = json.load(fp)
            data = pd.read_csv(csv_path, index_col=0)
            target = info_file['inputs']['data'][0]['targets'][0]['colName']
            x = data.drop(target, axis=1)
            y = data[target]
            yield (x, y, csv_path, info_path)


def run_pipeline(x, y, model_name, model):
    score = np.nan
    try:
        pipe = Pipeline([('scale', MinMaxScaler()), (model_name, model)])
        signal.alarm(180)
        score = np.mean(cross_validate(pipe, x, y, cv=3, error_score=-1)['test_score'])
        # time.sleep(10)
    except TimeoutError:
        score = TIMEOUT
    finally:
        signal.alarm(0)
    if score == -1:
        score = ERROR
    return score


classifiers = [
    ("knn", KNeighborsClassifier(3)),
    ('linear_svm', SVC(kernel="linear", C=0.025)),
    ('rbf_svm', SVC(gamma=2, C=1)),
    ('gausian_process', GaussianProcessClassifier(1.0 * RBF(1.0))),
    ('decision_tree', DecisionTreeClassifier(max_depth=5)),
    ('random_forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ('mlp', MLPClassifier(alpha=1)),
    ('ada_boost', AdaBoostClassifier()),
    ('naive_bayes', GaussianNB()),
    ('qda', QuadraticDiscriminantAnalysis())]


df = pd.read_csv("/users/guest/m/masonfp/Desktop/new_metafeatures/data/accuracy.csv")

for x, y, data_path, info_path in data_iterator():
    print(data_path)
    x = pd.get_dummies(x.fillna(x.mean()), dummy_na=True)
    for name, model in classifiers:
        instance = df[(df['data_path'] == data_path) & (df['classifier'] == name)]
        instance = instance.iloc[0] if len(instance) > 0 else None
        accuracy = pd.to_numeric(instance['accuracy'], errors='ignore') if instance is not None else None
        if instance is not None and accuracy not in [TIMEOUT, ERROR] and not np.isnan(accuracy):
            pass
        else:
            print(f"\t{name}: ", end='', flush=True)
            result = run_pipeline(x, y, name, model)
            print(f'{result}', flush=True)
            new_row = pd.Series({'data_path': data_path, 'info_path': info_path, 'classifier': name, 'accuracy': result})
            if instance is None:
                df = df.append(new_row, ignore_index=True)
            else:
                df.iloc[instance.name] = new_row
            df.to_csv("/users/guest/m/masonfp/Desktop/new_metafeatures/data/accuracy.csv", index=False)

print(df)
