import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Prediction on records
def predict_SVM():
    df = pd.read_csv('Voice.csv')
    # get 0-3 columns in jumps of 2
    X = df.iloc[:, 0:19].to_numpy()
    y = df.iloc[:, 19].to_numpy()

    sum_train = 0
    sum_test = 0

    # split the data into train and test
    X_train = X
    y_train = y

    df1 = pd.read_csv('voiceTest.csv')
    X_test = df1.iloc[:, 0:19]
    y_test = df1.iloc[:, 19]

    model = SVC(kernel="linear", C=6)
    model.fit(X_train, y_train)

    predict_test = model.predict(X_test)
    predict_train = model.predict(X_train)


    for i in range(len(predict_test)):
        if predict_test[i] == "male":
            predict_test[i] = "female"
        else:
            predict_test[i] = "male"

    sum_test += accuracy_score(y_test, predict_test)
    sum_train += accuracy_score(y_train, predict_train)

    if len(y_test) == 1:
        gender = y_test[0]
        print("Gender: {} ".format(predict_test), (predict_test == gender))
    else:
        print("Accuracy Test: ", sum_test)

    print("Accuracy Train: ", sum_train)


# Prediction on dataset Voice
def SVM():
    df = pd.read_csv('Voice.csv')
    # get 0-3 columns in jumps of 2
    X = df.iloc[:, 0:19].to_numpy()
    y = df.iloc[:, 19].to_numpy()

    sum_train = 0
    sum_test = 0

    for i in range(5):

        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = SVC(kernel="linear", C=6)
        model.fit(X_train, y_train)

        predict_test = model.predict(X_test)
        predict_train = model.predict(X_train)

        sum_test += accuracy_score(y_test, predict_test)
        sum_train += accuracy_score(y_train, predict_train)

    print("Accuracy Train: ", sum_train / 5)
    print("Accuracy Test: ", sum_test / 5)

