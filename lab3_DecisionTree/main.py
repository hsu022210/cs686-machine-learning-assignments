from decision_tree import decision_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("zoo.csv", header=None)
x_data = data.iloc[:, 1:-1].values.tolist()
y_data = data.iloc[:, -1].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=33)

criterions = ["entropy", "gini"]

for cr in criterions:
    dt = decision_tree(criterion=cr)
    dt.fit(x_train,y_train)
    hyp = dt.predict(x_test)
    print("====================================")
    print("Result for {0}:".format(cr))
    print("prediction:", hyp)
    print("y_test:", y_test)
    print("accuracy:", accuracy_score(hyp, y_test))
