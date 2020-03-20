import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def get_prediction(input_data, weights):
    return (weights * input_data).sum()


def get_slope(pred, target):
    slope = 2 * input_data * (pred - target)
    return slope


def get_mse(pred, target):
    mse = mean_squared_error(target, pred)
    return mse


if __name__ == "__main__":
    target = 2
    learning_rate = 0.01
    max_iterations = 100
    acceptation = 0.001

    input_data = np.array([5, 2, 1])
    weights = np.array([3, 0, -5])
    mse_list = []

    for i in range(max_iterations):
        pred = get_prediction(input_data, weights)
        slope = get_slope(pred, target)
        mse = get_mse([pred], [target])

        mse_list.append(mse)
        weights = weights - learning_rate * slope
        accuracy = abs(target - pred)
        if (accuracy < acceptation):
            print("Weights ", weights)
            break

    plt.plot(mse_list)
    plt.ylabel('Mean squared error')
    plt.xlabel('Iterations')
    plt.show()
