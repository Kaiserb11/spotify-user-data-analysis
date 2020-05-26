import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import pickle


file = "spotify.csv"
data = pd.read_csv(file, sep=',', header=0)
con = preprocessing.LabelEncoder()
predict = 'mode'

X = np.array(data.drop([predict], 1))
Y = np.array(con.fit_transform(list(data[predict])))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)


try:
    pickle_in = open("decision.pickle", "rb")
    clf_entropy = pickle.load(pickle_in)


except:
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=7, min_samples_leaf=4)
    clf_entropy.fit(X_train, Y_train)
    with open("decision.pickle", "wb") as f:
        pickle.dump(clf_entropy, f)



acc = clf_entropy.score(X_test, Y_test)
print(acc)

prediction = clf_entropy.predict(X_test)
for i in range(len(prediction)):
    print("predicted : ", prediction[i], '  ',"actual : ", Y_test[i])
