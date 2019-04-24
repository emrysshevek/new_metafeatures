import pandas as pd
import numpy as np

from metalearn import Metafeatures

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

datasets = pd.read_csv("/users/guest/m/masonfp/Desktop/new_metafeatures/data/accuracy.csv")
data = pd.read_csv(datasets.iloc[0][0], index_col=0)
mfs = Metafeatures().compute(data[data.columns[:-1]], data[data.columns[-1]], exclude=EXCLUDED)
mf_df = pd.DataFrame(mfs).iloc[1]
print(mf_df)
