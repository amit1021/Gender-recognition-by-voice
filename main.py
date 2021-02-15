from DecisionTree import predict_decision_tree, decision_tree
from ExtractFeatures import spectral_properties, predict_one, predict_folder
from KNN import predict_KNN, KNN
from LogisticRegression import pedict_logistic_regreesion, logistic_regreesion
from SVM import predict_SVM, SVM
from termcolor import colored


# Gender recognition by specific recording
def predict():
    print(colored("--------------------------------Logistic Regression--------------------------------", 'red'))
    pedict_logistic_regreesion()
    print(colored("-----------------------------------Decision Tree-----------------------------------", 'red'))
    predict_decision_tree()
    print(colored("----------------------------------------KNN----------------------------------------", 'red'))
    predict_KNN()
    print(colored("----------------------------------------SVM----------------------------------------", 'red'))
    predict_SVM()

# Gender recognition on dataset
def score_on_dataset():
    print(colored("--------------------------------Logistic Regression--------------------------------", 'red'))
    logistic_regreesion()
    print(colored("-----------------------------------Decision Tree-----------------------------------", 'red'))
    decision_tree()
    print(colored("----------------------------------------KNN----------------------------------------", 'red'))
    KNN()
    print(colored("----------------------------------------SVM----------------------------------------", 'red'))
    SVM()


if __name__ == '__main__':
    # Prediction on one recording
    predict_one("records/male/goodbye.wav", "male")
    predict()

    # Prediction on the recordings folder
    predict_folder()
    predict()

    # Prediction on dataset
    score_on_dataset()