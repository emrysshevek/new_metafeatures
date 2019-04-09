from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

from pandas import DataFrame

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection.base import SelectorMixin


# https://stackoverflow.com/a/3862957
def get_all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in get_all_subclasses(s)]


def sample_param_distributions(param_distributions):

    try:
        return sample_param_distributions_dict(param_distributions)
    except AttributeError:
        i = np.random.randint(len(param_distributions))
        return sample_param_distributions_dict(param_distributions[i])


def sample_param_distributions_dict(param_distributions_dict):

    params = {}
    for k, v in param_distributions_dict.items():
        i = np.random.randint(len(v))
        params[k] = v[i]
    return params


class AbstractParameterized(ABC):

    param_distributions = {}

    @classmethod
    def get_random_parameters(cls):
        return sample_param_distributions(cls.param_distributions)


class AbstractSupportsTask(ABC):

    @classmethod
    @abstractmethod
    def supports_task(cls, d3m):
        """Does this feature selector support the problem?
        :param d3m:
        :type d3m: D3M
        :return:
        :rtype: bool
        """
        pass


class AbstractSupportsVariable(ABC):

    @classmethod
    @abstractmethod
    def supports_variable(cls, d3m_var):
        """Does this feature selector support the problem?
        :param d3m:
        :type d3m: OrderedDict[str, D3MVariable]
        :return:
        :rtype: bool
        """
        pass


class AbstractFeatureExtractor(AbstractParameterized, AbstractSupportsVariable, BaseEstimator):

    def fit(self, df, variables):
        self.fit_transform(df, variables)
        return self

    @abstractmethod
    def fit_transform(self, df, variables):
        """ Fits the feature extractor
        :param df:
        :type df: DataFrame
        :param variables:
        :type variables: list[D3MVariable]
        :return:
        :rtype: csr_matrix
        """
        pass

    @abstractmethod
    def transform(self, df):
        """ Transforms the data
        :param df:
        :type df: DataFrame
        :return:
        :rtype: csr_matrix
        """
        pass


class AbstractFeatureSelector(AbstractParameterized, AbstractSupportsTask, BaseEstimator, SelectorMixin):

    pass


class AbstractEstimator(AbstractParameterized, BaseEstimator):

    @abstractmethod
    def fit(self, X, y):
        """
        :param X:
        :type X: csr_matrix
        :param y:
        :type y: ndarray
        :return:
        :rtype: AbstractEstimator
        """
        return self

    @abstractmethod
    def predict(self, X):
        """
        :param X:
        :type X: csr_matrix
        :return:
        :rtype: ndarray
        """
        pass


class AbstractMetric(ABC):

    # name of metric in schema, used for automatic lookup
    schema_name = 'abstractMetric'

    # whether lower is better (False) or higher is better (True)
    increasing = True

    def __init__(self, d3m):
        """
        :param d3m:
        :type d3m: D3M
        """
        self.d3m = d3m

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """
        :param y_true:
        :type y_true: list
        :param y_pred:
        :type y_pred: list
        :return:
        :rtype: float
        """
        pass