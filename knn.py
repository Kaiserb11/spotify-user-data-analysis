import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, neighbors
import pickle

data = pd.read_csv("spotify.csv")

predict = 'mode'
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

try:
    pickle_in = open("KNN.pickle", "rb")
    model = pickle.load(pickle_in)
    print("hi")

except:
    model = neighbors.KNeighborsClassifier(n_neighbors=6)
    with open("KNN.pickle", "wb") as f:
        pickle.dump(model, f)
    print("yo")
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

predicted = model.predict(x_test)

for x in range(len(predicted)):
    print("Predicted: ", predicted[x], "   Actual: ", y_test[x])

print(acc)
