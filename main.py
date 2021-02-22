from Adaboost import adaboost, predict_Adaboost
from DecisionTree import predict_decision_tree, decision_tree
from ExtractFeatures import spectral_properties, predict_one, predict_folder
from KNN import predict_KNN, KNN
from LogisticRegression import logistic_regreesion, predict_logistic_regreesion
from RandomForest import predict_random_forest, random_forest
from SVM import predict_SVM, SVM
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")


# Gender recognition by specific recording
def predict():
    print(colored("--------------------------------Logistic Regression--------------------------------", 'red'))
    predict_logistic_regreesion()
    print(colored("-----------------------------------Decision Tree-----------------------------------", 'red'))
    predict_decision_tree()
    print(colored("----------------------------------------KNN----------------------------------------", 'red'))
    predict_KNN()
    print(colored("----------------------------------------SVM----------------------------------------", 'red'))
    predict_SVM()
    print(colored("--------------------------------------Adaboost--------------------------------------", 'red'))
    predict_Adaboost()
    print(colored("--------------------------------------Random Forest--------------------------------------", 'red'))
    predict_random_forest()


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
    print(colored("--------------------------------------Adaboost--------------------------------------", 'red'))
    adaboost()
    print(colored("------------------------------------Random Forest------------------------------------", 'red'))
    random_forest()



if __name__ == '__main__':

    # Prediction on one recording
    # predict_one("records/male/lewis_prime-time_ratings.wav", "male")
    # predict()

    # Prediction on the recordings folder
    # predict_folder()
    # predict()

    # Prediction on dataset
    score_on_dataset()
