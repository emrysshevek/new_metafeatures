from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

CLASSIFIERS = {
    "knn": KNeighborsClassifier(3),
    'linear_svm': SVC(kernel="linear", C=0.025),
    'rbf_svm': SVC(gamma=2, C=1),
    'gausian_process': GaussianProcessClassifier(1.0 * RBF(1.0)),
    'decision_tree': DecisionTreeClassifier(max_depth=5),
    'random_forest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'mlp': MLPClassifier(alpha=1),
    'ada_boost': AdaBoostClassifier(),
    'naive_bayes': GaussianNB(),
    'qda': QuadraticDiscriminantAnalysis()}
