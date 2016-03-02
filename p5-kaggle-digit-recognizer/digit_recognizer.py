import numpy as np
import pandas as pd

from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA, FastICA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm, metrics, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import time

def getDataStats(data, desc):
    n_digits = data.shape[0]
    n_features = data.shape[1]

    print "Dataset {} --------".format(desc)
    print "Total number of digits (images): {}".format(n_digits)
    print "Total number of features: {}".format(n_features)

    # image is square
    image_width = image_height = np.ceil(np.sqrt(n_features)).astype(np.uint8)

    print "Digit image width: {}".format(image_width)
    print "Digit image height: {}".format(image_height)
    

direction_vectors1 = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]],

        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 1],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [1, 0, 0]],
        
        [[0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]]

def nudge_dataset(X, Y, direction_vectors):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 28x28 images in X around by 1px to left, right, down, up
    """
    start = time.time()
    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    
    Y = np.concatenate([Y for _ in range(13)], axis=0)
    end = time.time()

    print "Done!\nNudging time (secs): {:.3f}".format(end - start)
    
    return X, Y

def save_model(clf):
    joblib.dump(clf, "model.pkl")

    print "Model saved"

def load_model():
    clf = joblib.load('model.pkl')
    
    return clf

def plot_components(components, name):
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(components):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(str(components.shape[0])+' components extracted by '+name, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()

def draw_image(img):
    one_image = img.reshape(-1, 1, 28, 28)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

def min_max_scaler(X):
    scaler = MinMaxScaler(feature_range=(-1,1))
    start = time.time()
    X = scaler.fit_transform(X)
    end = time.time()
    
    print "Done!\nFit MinMaxScaler transform time (secs): {:.3f}".format(end - start)

    return X, scaler

def fit_transform_pca(X):
    start = time.time()
    pca = PCA(n_components=110, whiten=False) #110
    X = pca.fit_transform(X)
    end = time.time()
    
    print pca.explained_variance_ratio_
    print "Done!\nFit PCA transform time (secs): {:.3f}".format(end - start)

    return X, pca

def fit_transform_ica(X):
    ica = FastICA(n_components=50, max_iter=2000, tol=0.05, algorithm='parallel', fun='cube', fun_args={'alpha': 1.0}, random_state=42) #26 36 76
    start = time.time()
    X = ica.fit_transform(X)
    end = time.time()
    
    print "Done!\nFit ICA transform time (secs): {:.3f}".format(end - start)

    return X, ica

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
      % (clf, metrics.classification_report(target, y_pred, digits=5)))
    
    return y_pred

def predict_labels_final(clf, features):
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction final time (secs): {:.3f}".format(end - start)
    
    y_pred_frame = pd.DataFrame(data=y_pred)
    y_pred_frame.index +=1
    y_pred_frame.columns = ['Label']

    y_pred_frame.to_csv(path_or_buf='data/test_labels2.csv', sep=',', index=True, index_label='ImageId')
    print "Test data written!"
    

def grid_search(X_train, y_train, X_test, y_test):
    #Decision Tree Classifier #####################################
    ###############################################################
    #clf = DecisionTreeClassifier()
    #parameters = {'max_depth':(11,12,13), 'min_samples_split':(4,5), 'min_samples_leaf':(5,6)} #'splitter':('best', 'random') 'min_samples_leaf':(5,6)

    #K Neighbors Classifier #######################################
    ###############################################################
    #clf = KNeighborsClassifier()
    #parameters = {'n_neighbors':(3,4,5,6,7,8,9), 'algorithm':['kd_tree'], 'p':(2,3), 'weights':['distance']}

    #Support Vector Machines ######################################
    ###############################################################
    #clf = svm.SVC()
    parameters = {'C': (2,3,4), 'gamma': (0.009,0.008,0.007), 'kernel': ['rbf']}
    #parameters = {'C': (10,20,40), 'gamma': (0.01,0.001), 'kernel': ['poly'], 'degree': [9]}
    #parameters = {'C': (5,6), 'gamma': (0.007,0.005), 'kernel': ['poly'], 'degree': [2,3,4]}
    #parameters = {'C': (2,3,4,5,6,7), 'gamma': (0.015,0.01,0.007,0.005), 'kernel': ['rbf']}

    #Bernoulli RBM ################################################
    ###############################################################
    #rbm = BernoulliRBM()
    #logistic = linear_model.LogisticRegression()
    #parameters = {'rbm__learning_rate': [0.1, 0.01, 0.001], 'rbm__n_iter': [20, 40, 80], 'rbm__n_components': [50, 100, 200], 'rbm__random_state': [42],
    #              'logistic__C': [1.0, 10.0, 100.0]}
    #parameters = {'rbm__learning_rate': [0.01], 'rbm__n_iter': [20], 'rbm__n_components': [200], 'rbm__random_state': [42],
    #              'logistic__C': [10.0, 15.0]}
    #clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    #Grid Search ##################################################
    ###############################################################
    #SVC & LogisticRegression with BernoulliRBM
    clf = GridSearchCV(clf, parameters, cv=2, verbose=1)

    #All Other Models
    #p_metric = metrics.make_scorer(metrics.f1_score, average="weighted")
    #clf = GridSearchCV(clf, parameters, scoring=p_metric, cv=2, verbose=1)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining grid search time (secs): {:.3f}".format(end - start)

    y_pred = clf.predict(X_test)

    print "Best model parameters: "+str(clf.best_estimator_)
    print "Best model parameters: "+str(clf.best_params_)
    print "Best score: "+str(clf.best_score_)
    #print "Best score: "+str(clf.grid_scores_)
    
    print "Final F1 score for test: "+str(metrics.f1_score(y_test, y_pred, average='weighted'))


def run_test(X_train, y_train, X_test, y_test, test_data_final):

    #clf = GaussianNB()
    #clf = DecisionTreeClassifier(max_depth=14, splitter='best', min_samples_split=7, min_samples_leaf=5)
    #clf = KNeighborsClassifier(n_neighbors=4, algorithm="kd_tree", p=2, weights='distance', n_jobs=4)
    #clf = BaggingClassifier(knn, n_estimators=10, max_samples=1.0, max_features=1.0, random_state=42)
    clf = svm.SVC(kernel="rbf", C=3, gamma=0.008, cache_size=1000)
    #clf = svm.SVC(kernel="poly", C=2, gamma=0.01, cache_size=1000, degree=9)

    train_classifier(clf, X_train, y_train)

    print "Traing set: "
    predict_labels(clf, X_train, y_train)

    print "Test set: "
    y_pred = predict_labels(clf, X_test, y_test)

    print "Test set final: "
    predict_labels_final(clf, test_data_final)

def run_test_rbm(X_train, y_train, X_test, y_test, test_data_final):
    rbm = BernoulliRBM(learning_rate=0.01, n_iter=20, n_components=350, verbose=1, random_state=42) #learning_rate=0.001, n_iter=20, n_components=200, verbose=1, random_state=42

    #lr = linear_model.LogisticRegression(C=15.0)
    sv = svm.SVC(kernel="rbf", C=3, gamma=0.008, cache_size=1000)
    
    clf = Pipeline(steps=[('rbm', rbm), ('sv', sv)])

    train_classifier(clf, X_train, y_train)

    print "RBM Traing set: "
    predict_labels(clf, X_train, y_train)

    print "RBM Test set: "
    y_pred = predict_labels(clf, X_test, y_test)

    # Plot RBM components
    #plot_components(rbm.components_, 'RBM')

    # Logistic Regression Raw Data
    #logistic_classifier = linear_model.LogisticRegression()
    #logistic_classifier.fit(X_train, y_train)
    
    #print "LR Traing set: "
    #predict_labels(logistic_classifier, X_train, y_train)

    #print "LR Test set: "
    #y_pred = predict_labels(logistic_classifier, X_test, y_test)

def run(X_train, y_train, X_test, y_test, test_data_final):
    #grid_search(X_train, y_train, X_test, y_test)
    run_test(X_train, y_train, X_test, y_test, test_data_final)
    #run_test_rbm(X_train, y_train, X_test, y_test, test_data_final)


# get data
train_data = pd.read_csv("data/train.csv", nrows=1000) #set to nrows=42000 for full train test, note will take a few hours to run with nudge_dataset
print "Train data loaded!"

test_data_final = pd.read_csv("data/test.csv")
print "Test data loaded!"

feature_cols_all = list(train_data.columns[1:])
target_col_all = train_data.columns[0]

X_train = train_data[feature_cols_all]
y_train = train_data[target_col_all]

getDataStats(X_train, 'train')
#getDataStats(test_data_final, 'test')

# calculate unique labels with count (train dataset)
unique_features, unique_feature_count = np.unique(y_train, return_counts=True)
print "Unique labels with count:"
print np.asarray((unique_features, unique_feature_count))

#print X_train.head()
#print y_train.head()

# Expand the Data Set
X_train, y_train = nudge_dataset(X_train, y_train, direction_vectors1)
getDataStats(X_train, 'new train')

# calculate unique labels with count (train dataset)
unique_features, unique_feature_count = np.unique(y_train, return_counts=True)
print "Unique labels with count:"
print np.asarray((unique_features, unique_feature_count))

# Scale data
X_train, scaler = min_max_scaler(X_train)
#X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)

# Transform data
X_train, pca = fit_transform_pca(X_train)

# Plot PCA components
#plot_components(pca.components_, 'PCA')

# Plot PCA componet variance
#x = np.arange(200)
#plt.plot(x, 1 - np.cumsum(pca.explained_variance_ratio_), '-')
#plt.show()

#X_train, ica = fit_transform_ica(X_train)

#final test data
test_data_final = scaler.transform(test_data_final)
test_data_final = pca.transform(test_data_final)

X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

run(X_mtrain, y_mtrain, X_mtest, y_mtest, test_data_final)





