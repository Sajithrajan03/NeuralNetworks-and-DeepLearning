import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset and preprocess
iris = load_iris()
X = iris.data[:, [0, 3]]  # Using sepal length and petal width
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perceptron training algorithm

def perceptron_train(features, labels, learning_rate=0.1, epochs=100):
    num_samples, num_features = features.shape
    weights = np.random.rand(num_features)
    bias = np.random.rand()

    for epoch in range(epochs):
        for i in range(num_samples):
            prediction = np.dot(weights, features[i]) + bias
            if prediction >= 0:
                y_pred = 1
            else:
                y_pred = 0

            error = labels[i] - y_pred
            weights += learning_rate * error * features[i]
            bias += learning_rate * error

    return weights, bias

# Neural network classification

# Visualize the dataset and hyperplanes
def visualize_dataset(features, labels, weights_list, bias_list):
    plt.figure(figsize=(10, 6))

    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(features[indices, 0], features[indices, 1], label=label)

    for i, weights in enumerate(weights_list):
        bias = bias_list[i]
        slope = -weights[0] / weights[1]
        intercept = -bias / weights[1]

        x_vals = np.linspace(min(features[:, 0]), max(features[:, 0]), 100)
        y_vals = slope * x_vals + intercept

        plt.plot(x_vals, y_vals, label=f'Hyperplane {i+1}', linestyle='--')

    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.title('Iris Dataset with Hyperplanes')
    plt.show()

# Main code
if __name__ == "__main__":
    classes = np.unique(y_train)
    perceptron_weights = []
    perceptron_bias = []

    for c in classes:
        labels = np.where(y_train == c, 1, 0)
        weights, bias = perceptron_train(X_train, labels)
        perceptron_weights.append(weights)
        perceptron_bias.append(bias)

    # Visualize the dataset with hyperplanes
    visualize_dataset(X, y, perceptron_weights, perceptron_bias)

# Perceptron training algorithm


