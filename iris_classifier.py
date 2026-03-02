from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#step 2
iris = load_iris()
x = iris.data
y = iris.target


#step 3
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.2, random_state = 42)


#step 4
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

#step-5 : 
y_pred = clf.predict(x_test)

#step 6 :
priint("Accuracy :" , accuracy_score(y_test, y_pred))