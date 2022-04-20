"""
MODEL CLASS

should implement:

fit
predict
evaluate/score --> depending on a flag that's passed in, gets the:
    accuracy
    error (aka 1-acc)
    precision + recall
"""
from collections import Counter

import numpy


class SVMEstimator:
    def __init__(self, pegasos_lambda=2**-5, max_iterations=20):
        self.w = None
        self.pegasos_lambda = pegasos_lambda
        self.max_iterations = max_iterations

    def fit(self, x, y):
        w = numpy.zeros(len(x[0]))
        t = 0
        for iteration in range(0, self.max_iterations):
            for j in range(len(x)):
                t = t + 1
                stepsize = 1 / (t * self.pegasos_lambda)
                test_value = y[j] * (numpy.dot(w, x[j]))
                if test_value < 1:
                    w = ((1 - (stepsize * self.pegasos_lambda)) * w) + (stepsize * y[j] * x[j])
                else:
                    w = ((1 - (stepsize * self.pegasos_lambda)) * w)
        self.w = w

        return w

    def predict(self, x, w):
        return numpy.dot(w, x)

    def score(self, gt, preds, flag="accuracy"):
        if flag == "accuracy":
            k = 0
            for i in range(len(gt)):
                if gt[i] != preds[i]:
                    k += 1
            return [1 - (k / len(gt))]

        elif flag == "error":
            k = 0
            for i in range(len(gt)):
                if gt[i] != preds[i]:
                    k += 1
            return [(k / len(gt))]

        elif flag == "precision":
            true_pred = Counter()
            all_pred = Counter()
            precision_list = []

            for i in range(len(gt)):
                if gt[i] == preds[i]:
                    true_pred[preds[i]] += 1
                all_pred[preds[i]] += 1

            for key in true_pred.keys():
                curr_precision = (key, true_pred[key] / all_pred[key])
                precision_list.append(curr_precision)

            return precision_list

        elif flag == "recall":
            true_pred = Counter()
            all_val = Counter()
            recall_list = []

            for i in range(len(gt)):
                if gt[i] == preds[i]:
                    true_pred[preds[i]] += 1

                all_val[gt[i]] += 1

            for key in true_pred.keys():
                curr_precision = (key, true_pred[key] / all_val[key])
                recall_list.append(curr_precision)

            return recall_list


class SVM(SVMEstimator):
    def __init__(self, pegasos_lambda=2 ** -5, max_iterations=20, num_classes=10):
        super(SVM, self).__init__(pegasos_lambda, max_iterations)
        self.all_w = None
        self.w = None
        self.pegasos_lambda = pegasos_lambda
        self.max_iterations = max_iterations
        self.num_classes = num_classes
        self.predictions = None

    def onevsall(self, x, y):
        all_new_y_data = []
        all_w = []
        for i in range(self.num_classes):
            new_y_data = [0] * len(y)
            for j in range(len(y)):
                if y[j] == i:
                    new_y_data[j] = 1
                else:
                    new_y_data[j] = -1
            w = self.fit(x, new_y_data)
            all_new_y_data.append(new_y_data)
            all_w.append(w)
        all_w = numpy.array(all_w)

        self.all_w = all_w

        return all_w

    def predict_class(self, x):
        y_pred = numpy.zeros(len(x))
        for i in range(len(x)):
            all_preds = []
            for j in range(len(self.all_w)):
                pred = self.predict(x[i], self.all_w[j])
                all_preds.append(pred)
            y_pred[i] = numpy.argmax(all_preds)

        self.predictions = y_pred

        return y_pred

    def get_score(self, gt, pred, flag="accuracy"):
        return self.score(gt, pred, flag)



