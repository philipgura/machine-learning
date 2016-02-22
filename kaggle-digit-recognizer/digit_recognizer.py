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

test_data_final = None #pd.read_csv("data/test.csv")
print "Test data loaded!"

#data stats
n_digits = train_data.shape[0]
n_features = train_data.shape[1]

print "Total number of digits train: {}".format(n_digits)
print "Total number of features train: {}".format(n_features)

#n_digits = test_data_final.shape[0]
#n_features = test_data_final.shape[1]

#print "Total number of digits test: {}".format(n_digits)
#print "Total number of features test: {}".format(n_features)


#format data all
#train data
feature_cols_all = list(train_data.columns[1:])
target_col_all = train_data.columns[0]

X_train_all = train_data[feature_cols_all]
y_train_all = train_data[target_col_all]

#test data
#feature_cols_test_all = list(test_data.columns[1:])
#target_col_test_all = test_data.columns[0]

#X_test_all = test_data[feature_cols_test_all]
#y_test_all = test_data[target_col_test_all]

#print X_all.head()
#print y_all.head()

print "-----"

#format data mini block
mini_tdata = train_data[0:1000]

feature_cols_mini = list(mini_tdata.columns[1:])
target_col_mini = mini_tdata.columns[0]

X_mini = mini_tdata[feature_cols_mini]
y_mini = mini_tdata[target_col_mini]

#print X_mini.head()
#print y_mini.head()

#pca = PCA(n_components=50, whiten=False)
#X_mini = pca.fit_transform(X_mini)


ica = FastICA(n_components=26, max_iter=1000, tol=0.07, algorithm='parallel', fun='exp', random_state=42)
start = time.time()
X_mini = ica.fit_transform(X_mini)
end = time.time()
print "Done!\nFit transform time (secs): {:.3f}".format(end - start)

X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(X_mini, y_mini, test_size=0.2, random_state=42) #X_mini, y_mini

#X_train = X_train_all
#y_train = y_train_all
#X_test = X_test_all
#y_test = y_test_all


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

def predict_labels_final(clf, features):
    
    y_pred = clf.predict(features)

    y_pred_frame = pd.DataFrame(data=y_pred)
    y_pred_frame.index +=1
    y_pred_frame.columns = ['Label']

    y_pred_frame.to_csv(path_or_buf='data/test_labels.csv', sep=',', index=True, index_label='ImageId')
    print "Test data written!"
    

def grid_search(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier()
    
    p_metric = metrics.make_scorer(metrics.f1_score, average="weighted")
    parameters = {'max_depth':(11,12), 'min_samples_split':(4,5,6,7,8), 'min_samples_leaf':(5,6), 'splitter':('best', 'random')}
    clf = GridSearchCV(clf, parameters, scoring=p_metric, cv=10, n_jobs=6)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining grid search time (secs): {:.3f}".format(end - start)

    y_pred=clf.predict(X_test)

    print "Best model parameters: "+str(clf.best_params_)
    print "Best score: "+str(clf.best_score_)
    print "Final F1 score for test: "+str(metrics.f1_score(y_test, y_pred, average='weighted'))

def apply_pca(data):
    pca = PCA(n_components=50, whiten=False)
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


def run_test(X_train, y_train, X_test, y_test, test_data_final):

    #clf = svm.SVC()
    #clf = GaussianNB()



    clf = DecisionTreeClassifier(max_depth=14, splitter='best', min_samples_split=7, min_samples_leaf=5)
    
    #X_train = apply_pca(X_train)

    #X_train = apply_ica(X_train)
    #X_test = apply_ica(X_test)

    train_classifier(clf, X_train, y_train)

    print "Traing set: "
    predict_labels(clf, X_train, y_train)

    print "Test set: "
    predict_labels(clf, X_test, y_test)

    #print "Test set final: "
    #predict_labels_final(clf, test_data_final)

    


def run(X_train, y_train, X_test, y_test, test_data_final):
    grid_search(X_train, y_train, X_test, y_test)
    #run_test(X_train, y_train, X_test, y_test, test_data_final)

run(X_mtrain, y_mtrain, X_mtest, y_mtest, test_data_final)





