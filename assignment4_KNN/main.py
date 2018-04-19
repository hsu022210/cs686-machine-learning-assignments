import pandas as pd
import arff
import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import knn


def getXandY_from_arff(fileName):
    arffData = arff.load(open(fileName, 'r'))
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(len(arffData["attributes"]))

    attrs_list = []
    for attrMeta in arffData["attributes"]:
        attrs_list.append(attrMeta[0])

    df = pd.DataFrame(data=arffData['data'], columns=attrs_list)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def runKnn(X, y, start=2, end=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    for k in range(start, end+1):
        model = knn(k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print("k: " + str(k) + ", accuracy: " + str(accuracy_score(pred, y_test)))


if __name__ == "__main__":
    X, y = getXandY_from_arff("PhishingData.arff")
    runKnn(X, y, 2, 32)
