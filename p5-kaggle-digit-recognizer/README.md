## Kaggle - Digit Recognizer
The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is. Used: Python, sklearn, pandas, numpy

####Loading and Formatting Data
- Reading train_data and test_data from csv file
- Seperating train_data in to features and labels
- Creating a mini_tdata to work on a smaller subset of the traing data
- Split the mini_tdata in to testing and training sets

####Initial Algorithm Tests
Using PCA to compress the original dementions (784) to 50

Tryied Classifiers:
- DecissionTreeClassifier
- GaussianNB
- KNeighborsClassifier
- SVC
- LogisticRegression with BernoulliRBM

SVC looked the most promessing (using f1_score)

####Optimized Parameters
- Optimized PCA components to 110 vs original 50
- Used GridSearchCV to find the most optimal SVC parameters
- Ran GridSearchCV with full training data set to optimize SVC parameters more

####Final Run
Ran final test data: Nudge Dataset --> MinMaxScaler --> PCA --> SVC

`MinMaxScaler(feature_range=(-1.0, 1.0))`

`PCA(n_components=110, whiten=False)`

`svm.SVC(kernel="rbf", C=3, gamma=0.008, class_weight=weights, cache_size=1000)`

- Outputted the predicted label data to csv file
- Submitted results to Kaggle


