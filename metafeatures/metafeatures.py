import pandas as pd
import numpy as np

from metalearn import Metafeatures

from sklearn.decomposition import PCA


EXCLUDED = ['ClassProbabilities',
            'CardinalitiesOfCategoricalFeatures',
            'CardinalitiesOfNumericFeatures',
            'MeansOfNumericFeatures',
            'StdDevsOfNumericFeatures',
            'SkewnessesOfNumericFeatures',
            'KurtosesOfNumericFeatures',
            'CategoricalAttributeEntropies',
            'NumericAttributeEntropies',
            'CategoricalJointEntropies',
            'NumericJointEntropies',
            'CategoricalMutualInformation',
            'NumericMutualInformation',
            'MeansOfStringLengthOfTextFeatures',
            'StdDevsOfStringLengthOfTextFeatures',
            'SkewnessesOfStringLengthOfTextFeatures',
            'KurtosesOfStringLengthOfTextFeatures']


df = pd.read_csv("/users/guest/m/masonfp/Desktop/new_metafeatures/data/accuracy.csv")
cols = [*list(range(len(Metafeatures().list_metafeatures())-len(EXCLUDED))), 'classifier', 'accuracy']
metafeatures = pd.DataFrame(columns=cols)

for name, group in df.groupby('data_path'):
    print(name)
    # print(group)
    data = pd.read_csv(name, index_col=0)
    mfs = Metafeatures().compute(data[data.columns[:-1]], data[data.columns[-1]], exclude=EXCLUDED)
    mfs = pd.DataFrame(mfs).iloc[1].values
    print(mfs)

    classifiers = group['classifier']
    accuracies = group['accuracy']
    mf_df = pd.DataFrame([[*mfs, classifier, accuracy] for classifier, accuracy in zip(classifiers, accuracies)])
    metafeatures = metafeatures.append(mf_df)
    metafeatures.to_csv("metafeatures.csv", index=False)
