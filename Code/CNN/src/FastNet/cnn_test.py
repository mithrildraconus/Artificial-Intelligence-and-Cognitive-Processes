from CNN.src.FastNet.run import *
from CNN.src.FastNet.classifiers.convnet import *
import numpy as np

X_white_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/X_white_test_4000.pkl"
y_white_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/y_white_test_4000.pkl"
Xtest_white = get_data(X_white_path)
ytest_white = get_data(y_white_path)
X_black_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/X_black_test_4000.pkl"
y_black_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/y_black_test_4000.pkl"
Xtest_black = get_data(X_black_path)
ytest_black = get_data(y_black_path)

trained_models = load_models()

white_predictions = predict(Xtest_white, trained_models, predictPlayer)
black_predictions = predict(Xtest_black, trained_models, predictPlayer)
white_predict_acc = predictionAccuracy(white_predictions, ytest_white)
black_predict_acc = predictionAccuracy(black_predictions, ytest_black)

print(white_predictions)
print("White prediction accuracy: %s" % white_predict_acc)
print(black_predictions)
print("Black prediction accuracy: %s" % black_predict_acc)
