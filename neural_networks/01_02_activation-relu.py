import numpy as np


def relu(input):
    return max(input, 0)


def calc_node_output(row_data, weights):
    node_input = (row_data * weights).sum()
    node_output = relu(node_input)
    return node_output


if __name__ == "__main__":
    # Hidden layer nodes weights
    weights = {
        'node_0': np.array([1, 1]),
        'node_1': np.array([-1, 1]),
        'output': np.array([2, -1])
    }

    # Input
    input_data = np.array([2, 3])
    # Hidden layer noddes
    node_0 = calc_node_output(input_data, weights['node_0'])
    node_1 = calc_node_output(input_data, weights['node_1'])
    hidden_layer = np.array([node_0, node_1])
    # Output
    model_output = calc_node_output(hidden_layer, weights['output'])

    print(model_output)
