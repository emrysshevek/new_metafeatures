import os, sys, json, random, time, threading
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import re, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, mean_squared_error


here = os.path.dirname(os.path.abspath(__file__))

from d3mds import D3MDataset, D3MProblem, D3MDS
from feat import AnnotatedTabularExtractor
from estimation import SGDClassifierEstimator, SGDRegressorEstimator, RBFSamplerSGDClassifierEstimator, RBFSamplerSGDRegressorEstimator


dspath = os.path.join(here, '..', '..', 'LL0_uci_forest_fires_dataset')
prpath = os.path.join(here, '..', '..', 'LL0_uci_forest_fires_problem')
solpath = os.path.join(here, '..')
assert os.path.exists(dspath)
assert os.path.exists(prpath)

N_ITER = 100 # Number of parameter settings that are sampled in RandomizedSearchCV. n_iter trades off runtime vs quality of the solution.

d3mds = D3MDS(dspath, prpath) # create the D3MDS object from the dataset and problem paths

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)


class Estimator(BaseEstimator):

	def set_task_type(self, taskType):
		self.taskType = taskType
	
	def fit(self, X_train, y_train):
		print('... in fit method ...')
		self.X_train = X_train
		self.y_train = np.ravel(y_train)

		if os.path.exists(os.path.join(here, 'model.pkl')):
			self.model = joblib.load(os.path.join(here, 'model.pkl'))
		else:
			spinner = Spinner()
			spinner.start()
			if self.taskType == 'regression':
				sgdr = SGDRegressorEstimator()
				sgdr_est = RandomizedSearchCV(sgdr, sgdr.param_distributions, n_iter=N_ITER)
				sgdr_est.fit(self.X_train, self.y_train)
				sgdr_est_score = abs(sgdr_est.score(self.X_train, self.y_train))
				print(sgdr_est_score) # coefficient of determination R^2

				rbfr_est = RBFSamplerSGDRegressorEstimator()
				rbfr_est.sgdregressor = sgdr_est
				rbfr_est.fit(self.X_train, self.y_train)
				rbfr_estscore = abs(rbfr_est.score(self.X_train, self.y_train))
				print(rbfr_estscore) # coefficient of determination R^2

				if sgdr_est_score >= rbfr_estscore:
					self.model = sgdr_est
					self.modelScore = sgdr_est_score
				else:
					self.model = rbfr_est
					self.modelScore = rbfr_estscore
			elif self.taskType == 'classification':
				# still need to implement
				raise RuntimeError('Unimplemented task type !!!!')
			spinner.stop()
			joblib.dump(self.model, os.path.join(here, 'model.pkl'))


	def predict(self, X_test):
		print('... in predict method ...', X_test.shape)
		spinner = Spinner()
		spinner.start()
		self.X_test = X_test
		spinner.start()
		self.y_pred = self.model.predict(self.X_test)
		spinner.stop()
		return self.y_pred


def save_predictions(testData_index, y_pred):
	print('saving predictions ....')
	targetCols = []
	targets = d3mds.problem.get_targets()
	for target in targets: targetCols.append(target['colName'])
	print(targetCols)
	y_pred_df = pd.DataFrame(index=testData_index, data=y_pred, columns=targetCols)
	y_pred_df.to_csv(os.path.join(solpath, 'predictions.csv'))

def save_scores(metrics):
	df = pd.DataFrame(columns=['metric', 'value'])
	print('computing and saving scores on test data ....')
	y_true = d3mds.get_test_targets().ravel()
	for item in metrics:
		metric = item['metric']
		if metric == 'meanSquaredError':
			score = mean_squared_error(y_true, y_pred)
		elif metric == 'f1Macro':
			score = f1_score(y_true, y_pred, average='macro')
		elif metric == 'f1Micro':
			score = f1_score(y_true, y_pred, average='micro')
		else:
			raise RuntimeError('Unimplemented performance metric !!!!')

		print('score on test data: ', score)
		df.loc[len(df)] = [metric, score]
		df.to_csv('scores.csv')


if __name__ == '__main__':

	ate = AnnotatedTabularExtractor() # this is the feature extractor we are going to use

	# get some basic information from the problem and dataset schemas
	taskType = d3mds.problem.prDoc['about']['taskType']
	columns = d3mds.dataset.get_learning_data_columns()
	columnNames = [col['colName'] for col in columns]
	variables = dict(zip(columnNames, columns))
	metrics = d3mds.problem.prDoc['inputs']['performanceMetrics']
	
	# get the train data and perform featurization
	trainData = d3mds.get_train_data()
	print(trainData.shape)
	
	trainData = ate.fit_transform(trainData, variables)
	print(trainData.shape)

	trainTargets = d3mds.get_train_targets().ravel()
	print(trainTargets.shape)

	from sklearn.svm import SVR
	clf = SVR(C=1.0, epsilon=0.02)
	clf.fit(trainData, trainTargets)

	# get test data and perform featurization
	testData = d3mds.get_test_data()
	print(testData.shape)
	testData_index = testData.index
	testData = ate.transform(testData)
	print(testData.shape)
		
	y_pred = clf.predict(testData)
	save_predictions(testData_index, y_pred)

	save_scores(metrics)

	
	

	

	
