import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.datasets.samples_generator import make_classification
from sklearn import preprocessing
import sklearn.metrics
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import  validation_curve
from sklearn.model_selection import cross_val_score

from  sklearn.neighbors import KNeighborsClassifier
from  sklearn.linear_model import LinearRegression
from  sklearn.svm import SVC

#------------------------------iris dataset; KNN model

# iris = datasets.load_iris()
# iris_X = iris.data
# iris_Y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(
#     iris_X, iris_Y, test_size=0.3 )
#
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# print(knn.predict(X_test))
# print(y_test)
#
# plt.scatter(X_train, y_train)
# plt.show()

# -----------------------------------boston dataset; linear Regression model
# boston = datasets.load_boston()
# boston_X = boston.data
# boston_y = boston.target
# X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size= 0.3)
#
# linearModel = LinearRegression()
# linearModel.fit(X_train, y_train)
# print(linearModel.coef_)
# print(linearModel.intercept_)
# print(linearModel.score(X_train, y_train))
# print((linearModel.score(X_test,y_test)))


# -------------------------------------normolization SVM model

# X, y = make_classification(n_samples= 300, n_features= 2, n_redundant= 0,
#                            n_informative= 2, random_state= 22, n_clusters_per_class= 1
#                            ,scale= 100)
# X = preprocessing.scale(X)
#
# # plt.scatter(X[:, 0], X[:, 1], c = y)
# # plt.show()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
# model = SVC()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

# --------------------------------------- cross validation KNN model
# iris = datasets.load_iris()
# iris_X = iris.data
# iris_y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size= .3)
# k_scores = []
# k_range = range(1,31)
#
# for k in range(1,31):
#     model = KNeighborsClassifier(n_neighbors= k)
#     scores = cross_val_score(model, iris_X, iris_y, cv= 10, scoring= 'accuracy') #for classification
#     k_scores.append(scores.mean())
#
# plt.plot(k_range, k_scores)
# plt.xlabel(' Vlaue of K for KNN')
# plt.ylabel('Cross validation accuracy')
# plt.show()

# --------------------------------------------SVM, cross validation, learning curve
# digits = datasets.load_digits()
# X_digits = digits.data
# y_digits = digits.target
#
# # 用SVM学习并记录loss
# train_size, train_loss, test_loss = learning_curve(SVC(gamma=0.001), X_digits, y_digits,
#                                                    train_sizes=[0.1, 0.25,0.5,0.75,1],
#                                                    cv=10,scoring='accuracy')
# train_loss_mean = np.mean(train_loss,axis=1)
# test_loss_mean = np.mean(test_loss,axis=1)
#
# plt.plot(train_size, train_loss_mean, color= 'r',label='training')
# plt.plot(train_size,test_loss_mean, color='g', label='test')
# plt.xlabel('Training dataset')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# ----------------------SVM, validation curve to get the best value for gamma
#
# digit = datasets.load_digits()
# X_digit = digit.data
# y_digit = digit.target
# param_range = np.logspace(-6, -2, 5)
# train_loss, test_loss = validation_curve(SVC(), X_digit, y_digit,param_name='gamma',
#                                          param_range=param_range,cv=10,scoring='accuracy')
# train_loss_mean = np.mean(train_loss,axis=1)
# test_loss_mean = np.mean(test_loss,axis=1)
#
# plt.plot(param_range,train_loss_mean, color='r', label='Train')
# plt.plot(param_range,test_loss_mean, color='g', label='Test')
# plt.xlabel('Training dataset')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# -------------------------------read and write model

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

classifier = SVC()
classifier.fit(X_iris,y_iris)

# Save model
joblib.dump(classifier, 'save/SVM_classifier.pkl')
# restore model
classifier2 = joblib.load('save/SVM_classifier.pkl')
print(classifier2.predict(X_iris[:]))