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

KNeighborsClassifier looked the most promessing (using f1_score)

####Optimized Parameters
- Used GridSearchCV to find the most optimal KNeighborsClassifier parameters
- Ran GridSearchCV with full training data set to optimize KNeighborsClassifier parameters more.
- Used BaggingClassifier to take averages of KNeighborsClassifiers

####Final Run
Ran final test data thought the PCA --> BaggingClassifier --> KNeighborsClassifier

`PCA(n_components=50, whiten=False)`

`KNeighborsClassifier(n_neighbors=4, algorithm="kd_tree", p=2, weights='distance', n_jobs=4)`

`BaggingClassifier(knn, n_estimators=10, max_samples=1.0, max_features=1.0, random_state=42)`

- Outputted the predicted label data to csv file
- Submitted results to Kaggle


