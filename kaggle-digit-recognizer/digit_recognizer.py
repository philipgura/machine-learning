import numpy as np
import pandas as pd

from sklearn import svm, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, FastICA
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

import time

#get data
train_data = pd.read_csv("data/train.csv")
print "Train data loaded!"

#data stats
n_digits = train_data.shape[0]
n_features = train_data.shape[1]

print "Total number of digits: {}".format(n_digits)
print "Total number of features: {}".format(n_features)


#format data all
feature_cols_all = list(train_data.columns[1:])
target_col_all = train_data.columns[0]

X_all = train_data[feature_cols_all]
y_all = train_data[target_col_all]

#print X_all.head()
#print y_all.head()

print "-----"

#format data mini block
mini_tdata = train_data[0:30000]

feature_cols_mini = list(mini_tdata.columns[1:])
target_col_mini = mini_tdata.columns[0]

X_mini = mini_tdata[feature_cols_mini]
y_mini = mini_tdata[target_col_mini]

#print X_mini.head()
#print y_mini.head()


X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(X_all, y_all, test_size=0.3, random_state=42) #X_mini, y_mini


def train_classifier(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

def predict_labels(clf, features, target):
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(target, y_pred)))

def grid_search(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier()
    
    p_metric = metrics.make_scorer(metrics.f1_score, average="weighted")
    parameters = {'max_depth':(11,12,13,14,15), 'min_samples_split':(4,5,6,7,8,9,10), 'min_samples_leaf':(2,3,4,5,6), 'splitter':('best', 'random')}
    clf = GridSearchCV(clf, parameters, scoring=p_metric, cv=10)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining grid search time (secs): {:.3f}".format(end - start)

    y_pred=clf.predict(X_test)

    print "Best model parameters: "+str(clf.best_params_)
    print "Best score: "+str(clf.best_score_)
    print "Final F1 score for test: "+str(metrics.f1_score(y_test, y_pred, average='weighted'))

def apply_pca(data):
    pca = PCA(n_components=50)
    start = time.time()
    reduced_data = pca.fit_transform(data)
    end = time.time()
    print "Done!\nFit transform time (secs): {:.3f}".format(end - start)
    print pca.explained_variance_ratio_
    return reduced_data

def apply_ica(data):
    ica = FastICA(n_components=100, max_iter=1000, random_state=42)
    start = time.time()
    tra_data = ica.fit_transform(data)
    end = time.time()
    print "Done!\nFit transform time (secs): {:.3f}".format(end - start)
    return tra_data

def run_test(X_train, y_train, X_test, y_test):
    
    #clf = GaussianNB()
    clf = DecisionTreeClassifier(max_depth=14, splitter='best', min_samples_split=7, min_samples_leaf=5)
    
    #X_train = apply_pca(X_train)
    #X_test = apply_pca(X_test)

    #X_train = apply_ica(X_train)
    #X_test = apply_ica(X_test)

    train_classifier(clf, X_train, y_train)

    print "Traing set: "
    predict_labels(clf, X_train, y_train)

    print "Test set: "
    predict_labels(clf, X_test, y_test)


def run(X_train, y_train, X_test, y_test):
    #grid_search(X_train, y_train, X_test, y_test)
    run_test(X_train, y_train, X_test, y_test)

run(X_mtrain, y_mtrain, X_mtest, y_mtest)





