import numpy as np
import pandas as pd

from sklearn import svm, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, FastICA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

import time

#scaler = StandardScaler()
#X_mini = scaler.fit_transform(X_mini)

def getDataStats(data):
    n_digits = data.shape[0]
    n_features = data.shape[1]

    print "Total number of digits: {}".format(n_digits)
    print "Total number of features: {}".format(n_features)


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
      % (clf, metrics.classification_report(target, y_pred, digits=4)))
    
    return y_pred

def predict_labels_final(clf, features):

    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction final time (secs): {:.3f}".format(end - start)
    
    y_pred_frame = pd.DataFrame(data=y_pred)
    y_pred_frame.index +=1
    y_pred_frame.columns = ['Label']

    y_pred_frame.to_csv(path_or_buf='data/test_labels4.csv', sep=',', index=True, index_label='ImageId')
    print "Test data written!"
    

def grid_search(X_train, y_train, X_test, y_test):
    #clf = DecisionTreeClassifier()
    clf = KNeighborsClassifier()
    
    p_metric = metrics.make_scorer(metrics.f1_score, average="weighted")
    #parameters = {'max_depth':(11,12,13), 'min_samples_split':(4,5), 'min_samples_leaf':(5,6)} #'splitter':('best', 'random') 'min_samples_leaf':(5,6)
    parameters = {'n_neighbors':(9,10,11)} #'algorithm':('kd_tree', 'ball_tree'), 'p':(2,3), 'weights':('distance', 'uniform')
    clf = GridSearchCV(clf, parameters, scoring=p_metric, cv=2, n_jobs=1)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining grid search time (secs): {:.3f}".format(end - start)

    y_pred = clf.predict(X_test)

    print "Best model parameters: "+str(clf.best_params_)
    print "Best score: "+str(clf.best_score_)
    print "Final F1 score for test: "+str(metrics.f1_score(y_test, y_pred, average='weighted'))

def apply_pca(data):
    pca = PCA(n_components=100, whiten=False) #28 45 76 784
    start = time.time()
    reduced_data = pca.fit_transform(data)
    end = time.time()
    print "Done!\nFit PCA transform time (secs): {:.3f}".format(end - start)
    print pca.explained_variance_ratio_
    return reduced_data

def apply_ica(data):
    ica = FastICA(n_components=100, max_iter=1000, tol=0.01, algorithm='parallel', fun='exp', random_state=42) #26 36 76
    start = time.time()
    tra_data = ica.fit_transform(data)
    end = time.time()
    print "Done!\nFit ICA transform time (secs): {:.3f}".format(end - start)
    return tra_data


def run_test(X_train, y_train, X_test, y_test, test_data_final):

    #clf = svm.SVC(kernel="rbf", C=100, gamma=0.001)
    #clf = GaussianNB()
    clf = KNeighborsClassifier(n_neighbors=13, algorithm="kd_tree", leaf_size=30, p=3, weights='distance', n_jobs=8)

    #clf = DecisionTreeClassifier(max_depth=14, splitter='best', min_samples_split=7, min_samples_leaf=5)

    train_classifier(clf, X_train, y_train)

    print "Traing set: "
    predict_labels(clf, X_train, y_train)

    print "Test set: "
    y_pred = predict_labels(clf, X_test, y_test)

    #print "Test set final: "
    #predict_labels_final(clf, test_data_final)


def run(X_train, y_train, X_test, y_test, test_data_final):
    grid_search(X_train, y_train, X_test, y_test)
    #run_test(X_train, y_train, X_test, y_test, test_data_final)


#get data
train_data = pd.read_csv("data/train.csv")
print "Train data loaded!"

test_data_final = None #pd.read_csv("data/test.csv")
print "Test data loaded!"

getDataStats(train_data)

feature_cols_all = list(train_data.columns[1:])
target_col_all = train_data.columns[0]

X_train_all = train_data[feature_cols_all]
y_train_all = train_data[target_col_all]

#print X_all.head()
#print y_all.head()

#format data mini block
mini_tdata = train_data[0:500]

feature_cols_mini = list(mini_tdata.columns[1:])
target_col_mini = mini_tdata.columns[0]

X_mini = mini_tdata[feature_cols_mini]
y_mini = mini_tdata[target_col_mini]

#X_mini = apply_pca(X_mini)
#X_mini = apply_ica(X_mini)

start = time.time()
pca = PCA(n_components=28, whiten=False) #50
X_mini = pca.fit_transform(X_mini)
end = time.time()
print "Done!\nFit PCA transform time (secs): {:.3f}".format(end - start)


#ica = FastICA(n_components=28, max_iter=6000, tol=0.09, algorithm='parallel', fun='exp', fun_args={'alpha': 1.0}, random_state=42) #26 36 76
#start = time.time()
#X_mini = ica.fit_transform(X_mini)
#end = time.time()
#print "Done!\nFit ICA transform time (secs): {:.3f}".format(end - start)

#final test data
#test_data_final = apply_pca(test_data_final)
#test_data_final = apply_ica(test_data_final)
#test_data_final = pca.transform(test_data_final)

X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(X_mini, y_mini, test_size=0.1, random_state=42) #X_mini, y_mini

run(X_mtrain, y_mtrain, X_mtest, y_mtest, test_data_final)





