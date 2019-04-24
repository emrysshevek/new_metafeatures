import json
import argparse

import pandas as pd
import numpy as np

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

from classifiers import CLASSIFIERS

import warnings
warnings.filterwarnings("ignore")

TIMEOUT = 'TIMEOUT'
ERROR = 'ERROR'

df = pd.read_csv("data/accuracy.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    parser.add_argument("info", type=str)
    parser.add_argument("classifier", type=str)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    with open(args.info, 'r') as fp:
        info_file = json.load(fp)

    data = pd.read_csv(args.csv, index_col=0)
    target = info_file['inputs']['data'][0]['targets'][0]['colName']

    x = data.drop(target, axis=1)
    x = pd.get_dummies(x.fillna(x.mean()), dummy_na=True)
    y = data[target]

    instance = df[(df['data_path'] == args.csv) & (df['classifier'] == args.classifier)]
    instance = instance.iloc[0] if len(instance) > 0 else None
    accuracy = pd.to_numeric(instance['accuracy'], errors='ignore') if instance is not None else None

    if instance is None:
        print(f"\t{args.classifier}: ", end='', flush=True)
        pipe = Pipeline([('scale', MinMaxScaler()), (args.classifier, CLASSIFIERS[args.classifier])])
        result = np.mean(cross_validate(pipe, x, y, cv=3, error_score=np.nan, n_jobs=1)['test_score'])
        result = ERROR if np.isnan(result) else result
        print(f'{result}', flush=True)
        new_row = pd.Series({'data_path': args.csv, 'info_path': args.info, 'classifier': args.classifier, 'accuracy': result})
        if instance is None:
            df = df.append(new_row, ignore_index=True)
        else:
            df.iloc[instance.name] = new_row
        df.to_csv("data/accuracy.csv", index=False)
