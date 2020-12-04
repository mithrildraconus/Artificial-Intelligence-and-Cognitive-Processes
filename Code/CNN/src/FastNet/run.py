import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
from CNN.src.FastNet.layers import *
from CNN.src.FastNet.fast_layers import *
from CNN.src.FastNet.classifiers.convnet import *
from CNN.src.FastNet.classifier_trainer import ClassifierTrainer
from CNN.src.FastNet.gradient_check import eval_numerical_gradient_array
from CNN.src.FastNet.util import *


def get_data(path):
    f = open(path, 'rb')
    return pickle.load(f)


def initModels(fn):
    alekhine = fn()
    botvinnik = fn()
    capablanca = fn()
    euwe = fn()
    karpov = fn()
    kasparov = fn()
    petrosian = fn()
    smyslov = fn()
    spassky = fn()
    tal = fn()
    baseModel = fn()
    models = {'Alekhine': alekhine, 'Botvinnik': botvinnik, 'Capablanca': capablanca, 'Euwe': euwe, 'Karpov': karpov,
              'Kasparov': kasparov, 'Petrosian': petrosian, 'Smyslov': smyslov, 'Spassky': spassky, 'Tal': tal,
              'Other': baseModel}
    return models


def save_data(data, name):
    output = open(name, 'wb')
    pickle.dump(data, output)
    output.close()


def train(X_train, y_train, X_val, y_val, model, fn):
    trainer = ClassifierTrainer()
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
        X_train, y_train, X_val, y_val, model, fn,
        reg=0.0000, learning_rate=0.0015, batch_size=250, num_epochs=15,
        learning_rate_decay=0.999, update='sgd', verbose=True, dropout=1.0)

    return (best_model, loss_history, train_acc_history, val_acc_history)


def plot(net, loss_history, train_acc_history, val_acc_history):
    plt.suptitle("%s network results" % net)
    plt.subplot(2, 1, 1)
    plt.plot(train_acc_history)
    plt.plot(val_acc_history)
    plt.title('accuracy vs time')
    plt.legend(['train', 'val'], loc=4)
    plt.xlabel('epoch')
    plt.ylabel('classification accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss_history)
    plt.title('loss vs time')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()


def load_models():
    model_names = ['Other', 'Alekhine', 'Botvinnik', 'Capablanca', 'Euwe', 'Karpov', 'Kasparov', 'Petrosian', 'Smyslov', 'Spassky', 'Tal']
    names = ['other', 'alekhine', 'botvinnik', 'capablanca', 'euwe', 'karpov', 'kasparov', 'petrosian', 'smyslov', 'spassky', 'tal']
    trained_models = []

    for index, model_name in enumerate(model_names):
        path = 'C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/FastNet/%s_model.pkl' % model_name
        trained_model = get_data(path)
        trained_model = np.array(list(map(names.index, trained_model)))
        trained_models[names[index]] = trained_model
    return trained_models

def gradient_check(X, model, y):
    loss, grads = three_layer_convnet(X, model, y)
    dx_num = eval_numerical_gradient_array(lambda X: chess_convnet(X, model)[1]['W1'], X, grads)
    return rel_error(dx_num, grads['W1'])


def predict(X, model, fn):
    return fn(X, model)


def predictionAccuracy(predictions, label):
    return np.mean(predictions == label)


def scoreToCoordinateIndex(score):
    return (score // 8, score % 8)


def scoresToBoard(scores):
    return scores.reshape((8, 8))


def boardToScores(board):
    return board.reshape((64))


def predictPlayer(img, models):
    modelScores = {}
    scores = three_layer_convnet(img, models[0])
    for key in list(models.keys()):
        if key != 'Other':
            modelScores[key] = three_layer_convnet(img, models[key])

    availablePiecesBoard = clip_pieces(scores, img)  # (1, 64) size

    maxScore = 0
    maxChamp = None
    for i in range(64):
        coordinate = scoreToCoordinateIndex(i)
        if availablePiecesBoard[i] != 0:
            champType = INDEX_TO_CHAMP[np.argmax(img[:, coordinate[0], coordinate[1]])]
            availableMovesBoard = clip_moves(modelScores[champType], img, coordinate)
            composedScore = np.max(boardToScores(availableMovesBoard)) * availablePiecesBoard[i]
            if composedScore > maxScore:
                maxScore = composedScore
                maxChamp = scoreToCoordinateIndex(np.argmax(boardToScores(availableMovesBoard)))


    return maxChamp


def main():
    pass


if __name__ == "__main__":
    main()
