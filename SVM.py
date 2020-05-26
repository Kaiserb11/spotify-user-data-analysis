import sklearn
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle

data = pd.read_csv("spotify.csv")

predict = 'mode'
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

try:
    pickle_in  =open("SVM.pickle", "rb" )
    clf = pickle.load(pickle_in)
    
except:
    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)
    with open("SVM.pickle", "wb") as f:
        pickle.dump(clf, f)


y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

predicted = clf.predict(x_test)

for x in range(len(predicted)):
    print("Predicted: ", predicted[x], "   Actual: ", y_test[x])
print(acc)
