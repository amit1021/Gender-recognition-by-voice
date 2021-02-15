import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def pedict_logistic_regreesion():
    df = pd.read_csv('voice.csv')
    # get 0-3 columns in jumps of 2
    X = df.iloc[:, 0:19].to_numpy()
    y = df.iloc[:, 19].to_numpy()

    X_train = X
    y_train = y

    df1 = pd.read_csv('voiceTest.csv')
    X_test = df1.iloc[:, 0:19]
    y_test = df1.iloc[:, 19]

    clf = LogisticRegression(random_state=0, max_iter=3600).fit(X_train, y_train)
    clf.predict(X_test)
    clf.predict_proba(X_test)

    predict = clf.predict(X_test)

    for i in range(len(predict)):
        if predict[i] == "male":
            predict[i] = "female"
        else:
            predict[i] = "male"

    if len(y_test) == 1:
        gender = y_test[0]
        print("Gender: {} ".format(predict), (predict == gender))
    else:
        print("Score Test: {} ".format(1 - clf.score(X_test, y_test)))

    print("Score Train: {} ".format(clf.score(X_train, y_train)))


def logistic_regreesion():
    df = pd.read_csv('voice.csv')
    # get 0-3 columns in jumps of 2
    X = df.iloc[:, 0:19].to_numpy()
    y = df.iloc[:, 19].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression(random_state=0, max_iter=3600).fit(X_train, y_train)
    clf.predict(X_test)
    clf.predict_proba(X_test)

    print("Score Train: {} ".format(clf.score(X_train, y_train)))
    print("Score Test: {} ".format(clf.score(X_test, y_test)))

