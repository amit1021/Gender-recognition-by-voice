import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prediction on records
def predict_random_forest():
    # The features of the train
    df = pd.read_csv('voice.csv')
    # get 0-10 columns
    X = df.iloc[:, 0:11].to_numpy()
    # get 11 column
    y = df.iloc[:, 11].to_numpy()

    # split the data into train and test
    X_train = X
    y_train = y

    # The features of the recording
    df1 = pd.read_csv('voiceTest.csv')
    X_test = df1.iloc[:, 0:11]
    y_test = df1.iloc[:, 11]

    clf = RandomForestClassifier()
    clf = clf.fit(X_train, y_train)

    predict = clf.predict(X_test)

    if len(y_test) == 1:
        gender = y_test[0]
        print("Gender: {} ".format(predict), (predict == gender))
    else:
        print("Score Test: {} ".format(clf.score(X_test, y_test)))

    print("Score Train: {} ".format(clf.score(X_train, y_train)))


def random_forest():
    df = pd.read_csv('voice.csv')
    # get 0-10 columns
    X = df.iloc[:, 0:11].to_numpy()
    # get 11 column
    y = df.iloc[:, 11].to_numpy()

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print("Train" , clf.score(X_train ,y_train))
    print("Test",clf.score(X_test ,y_test))
