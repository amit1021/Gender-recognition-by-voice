import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Prediction on records
def pedict_logistic_regreesion():
    # The features of the train
    df = pd.read_csv('Voice.csv')
    # get 0-18 columns
    X = df.iloc[:, 0:19].to_numpy()
    # get 19 column
    y = df.iloc[:, 19].to_numpy()

    X_train = X
    y_train = y

    # The features of the recording
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

# Prediction on dataset Voice
def logistic_regreesion():
    # read Voice file
    df = pd.read_csv('Voice.csv')
    # get 0-18 columns
    X = df.iloc[:, 0:19].to_numpy()
    # get 19 column
    y = df.iloc[:, 19].to_numpy()
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create Logistic Regression classifier object
    clf = LogisticRegression(random_state=0, max_iter=3600).fit(X_train, y_train)
    clf.predict(X_test)
    clf.predict_proba(X_test)

    print("Score Train: {} ".format(clf.score(X_train, y_train)))
    print("Score Test: {} ".format(clf.score(X_test, y_test)))

