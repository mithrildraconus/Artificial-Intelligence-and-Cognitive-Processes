import numpy as np
import matplotlib.pyplot as plt
import caffe
import argparse


def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("data")
	parser.add_argument("training", type=bool)
	parser.add_argument("labels")
	parser.add_argument("net", choices=['Other', 'Alekhine', 'Botvinnik', 'Capablanca', 'Euwe', 'Karpov', 'Kasparov', 'Petrosian', 'Smyslov', 'Spassky',  'Tal'])
	parser.add_argument("gpu", type=bool)
	return parser.parse_args()


args = getArgs()
data = np.load(args.data, allow_pickle=True)

if args.training:  # Training

	print(("Training on %d inputs." % data.shape[0]))

	labels = np.load(args.labels, allow_pickle=True)
	solver = caffe.SGDSolver('solver_%s.prototxt' % args.net)
	solver.net.set_input_arrays(data, labels)
	solver.solve()

	print ("Training complete")

else:  # Testing

	print(("Testing on %d inputs." % inputs.shape[0]))

	classifier = caffe.Classifier("champ.prototxt", "%s_train.caffemodel" % args.net, gpu=args.gpu)
	prediction = classifier.predict(data)

	if args.labels:
		print(("Accuracy is %f" % np.mean(prediction == args.labels)))
	print(prediction)
