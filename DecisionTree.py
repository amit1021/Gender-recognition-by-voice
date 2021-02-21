from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split

# Prediction on records
def predict_decision_tree():
    # The features of the train
    df = pd.read_csv('voice.csv')
    # get 0-18 columns
    X = df.iloc[:, 0:10].to_numpy()
    # get 19 column
    y = df.iloc[:, 10].to_numpy()

    # split the data into train and test
    X_train = X
    y_train = y

    # The features of the recording
    df1 = pd.read_csv('voiceTest.csv')
    X_test = df1.iloc[:, 0:10]
    y_test = df1.iloc[:, 10]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    predict = clf.predict(X_test)

    for i in range(len(predict)):
        if predict[i] == "male":
            predict[i] = "female"
        else:
            predict[i] = "male"


    # we test one record
    if len(y_test) == 1:
        gender = y_test[0]
        print("Gender: {} ".format(predict), (predict == gender))
    else:
        # We test the folder of records
        print("Score Test: {} ".format(1 - clf.score(X_test, y_test)))

    print("Score Train: {} ".format(clf.score(X_train, y_train)))


# Prediction on dataset Voice
def decision_tree():

    df = pd.read_csv('voice.csv')
    # get 0-18 columns
    X = df.iloc[:, 0:10].to_numpy()
    # get 19 column
    y = df.iloc[:, 10].to_numpy()

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    print("Score Train: {} ".format(clf.score(X_train, y_train)))
    print("Score Test: {} ".format(clf.score(X_test, y_test)))

