from sklearn.datasets import load_iris
from sklean.model_selection import train_test_split
from sklearn.Tree import DecisionTreeClassfier
from metrics import accuracy_score

#step 2
iris = load_iris()
x = iris.data
y = iris.target


#step 3
(x_train, x_test, y_train, y_test) = train_test_split(test_size = 0.2, randome_state = 42)


#step 4
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

#step-5 : 
y_pred = clf.predict(x_test)