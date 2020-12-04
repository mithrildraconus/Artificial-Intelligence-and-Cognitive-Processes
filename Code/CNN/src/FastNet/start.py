from CNN.src.FastNet.run import *
from CNN.src.FastNet.classifiers.convnet import *
import numpy as np

init_convnet = init_three_layer_convnet
convnet = three_layer_convnet

# init_convnet = init_conv_convnet
# convnet = conv_convnet

netsToTrain = ['Other', 'Alekhine', 'Botvinnik', 'Capablanca', 'Euwe', 'Karpov', 'Kasparov', 'Petrosian', 'Smyslov', 'Spassky',  'Tal']

models = initModels(init_convnet)
for net in netsToTrain:
    nnet = netsToTrain.index(net)
    if nnet == 0:
        X_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/X_train_12000.pkl"
        y_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/y_train_12000.pkl"
        X = get_data(X_path)
        y = get_data(y_path)

        # X_w_train = X[:1000]
        # y_train = y[:1000]
        # X_val = X[1000:1100]
        # y_val = y[1000:1100]

        X_train = X[:10000]
        y_train = y[:10000]
        X_val = X[10000:12000]
        y_val = y[10000:12000]
    else:
        X_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/p%d_X_12000.pkl" % nnet
        y_path = "C:/Users/mithr/Desktop/Data Science Masters/DSC 680/Project 1/CNN/src/p%d_y_12000.pkl" % nnet
        X = get_data(X_path)
        y = get_data(y_path)
        s = int(5 * X.shape[0] / 6)
        X_train = X[:s]
        y_train = y[:s]
        X_val = X[s:]
        y_val = y[s:]

    y_val = np.array(list(map(netsToTrain.index, y_val)))
    y_train = np.array(list(map(netsToTrain.index, y_train)))

    results = train(X_train, y_train, X_val, y_val, models[net], convnet)
    best_model = np.array(results[0])

    output = open('%s_model.pkl' % net, 'wb')
    pickle.dump(best_model, output)
    output.close()

    plot(net, results[1], results[2], results[3])

