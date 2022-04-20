import numpy
from sklearn.model_selection import KFold
from model import SVM
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, "r") as fh:
        content = fh.readlines()
    return content


def normalize_data(content):
    total_vectors = []
    y_data = []

    for i in range(len(content)):
        label = int(content[i][0])
        y_data.append(label)
        vector_content = content[i][2:]
        vector = vector_content.split(",")
        for k in range(len(vector)):
            vector[k] = ((2 * int(vector[k])) / 255) - 1
        total_vectors.append(vector)

    return numpy.array(total_vectors), numpy.array(y_data)

def cross_validation(content, k, estimator):
    fold = KFold(n_splits=k)
    validation_errs = []

    content = numpy.array(content)

    for train_index, test_index in fold.split(content):  
        train_data = content[train_index]  
        test_data = content[test_index]  

        x_train, y_train = normalize_data(train_data)
        x_val, y_val = normalize_data(test_data)

        all_w = estimator.onevsall(x_train, y_train)
        val_pred = estimator.predict_class(x_val)
        val_err = estimator.get_score(y_val, val_pred, "error")
        validation_errs.append(val_err)

    return sum(validation_errs) / k


def main():
    filename = "mnist_train.txt"

    content = read_data(filename)
    # train = content[:1750]  # [784, 1]
    # validate = content[1750:]
    # x_train, y_train = normalize_data(train)
    # x_val, y_val = normalize_data(validate)
    # print("done building training examples")

    # pegasos_lambda = 2 ** -5
    #
    # svm = SVM(pegasos_lambda, 20, 10)
    # all_w = svm.onevsall(x_train, y_train)
    # print(all_w)
    # predictions = svm.predict_class(x_val)
    # print(predictions)
    # accuracy = svm.get_score(y_val, predictions, "accuracy")
    # print(f"accuracy = {accuracy}")
    # error = svm.get_score(y_val, predictions, "error")
    # print(f"error = {error}")
    # precision = svm.get_score(y_val, predictions, "precision")
    # print(f"precision = {precision}")
    # recall = svm.get_score(y_val, predictions, "recall")
    # print(f"recall = {recall}")

    # k = 5
    # cross_val_error = cross_validation(content, k, svm)
    # print(f"cross_val_error = {cross_val_error}")

    # pegasos_lambdas = [2**i for i in range(-5, 2)]
    # svms = [SVM(p) for p in pegasos_lambdas]
    #
    # avg_errors = []
    # for j in range(len(svms)):
    #     avg_err = cross_validation(content, k, svms[j])
    #     avg_errors.append(avg_err)
    #     print(f"ran {pegasos_lambdas[j]}, error: {avg_err}")
    #
    # log_pl = list(range(-5, 2))
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.scatter(log_pl, avg_errors, color="pink")
    # plt.xlabel("lambda")
    # plt.ylabel("error")
    # plt.title("cross val")
    # plt.show()
    #
    # min_error = min(avg_errors)
    # min_lambda = pegasos_lambdas[numpy.argmin(avg_errors)]
    # print(min_lambda)

    pegasos_lambda = 2 ** -3
    x_train, y_train = normalize_data(content)
    svm = SVM(pegasos_lambda)
    all_w = svm.onevsall(x_train, y_train)
    train_pred = svm.predict_class(x_train)
    train_err = svm.get_score(y_train, train_pred, "recall")
    print(f"train_err = {sorted(train_err, key=lambda x: x[0])}")

    filename = "mnist_test.txt"
    test = read_data(filename)
    x_test, y_test = normalize_data(test)
    test_pred = svm.predict_class(x_test)
    test_err = svm.get_score(y_test, test_pred, "recall")
    print(f"test_err = {sorted(test_err, key=lambda x: x[0])}")


main()
