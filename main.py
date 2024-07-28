import numpy as np
from NeuralNetwork import NeuralNetwork
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


######################## first DataSet ########################

def csv_to_np(str_file_name):
    """
    Convert a CSV file to numpy arrays for features and labels.

    Args:
        str_file_name (str): Path to the CSV file.

    Returns:
        tuple: Two elements - features (X) and labels (y) as numpy arrays.
    """
    df = pd.read_csv(str_file_name)
    X = np.array(df.iloc[:, :-1].values)
    y = np.array(df.iloc[:, -1].values)
    return X, y


def load_first_dataset():
    """
    Load the MNIST dataset and normalize the pixel values.

    Returns:
        tuple: Four elements - training features, training labels, test features, test labels.
    """
    X_train, y_train = csv_to_np("MNIST-train.csv")
    X_test, y_test = csv_to_np("MNIST-test.csv")

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape and convert to appropriate structure
    X_train = [x.reshape(-1, 1) for x in X_train]
    X_test = [x.reshape(-1, 1) for x in X_test]
    y_train = [vectorized_mnist_result(y) for y in y_train]
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def vectorized_mnist_result(j):
    """
    Convert a digit (0-9) into a one-hot encoded vector.

    Args:
        j (int): The digit to be converted.

    Returns:
        numpy.ndarray: A 10x1 one-hot encoded vector.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def main_first_dataset():
    """
    The main function that loads the data, initializes the neural network,
    trains it on the training data, and evaluates its performance on the test data.
    """
    # Load the data
    X_train, y_train, X_test, y_test = load_first_dataset()

    # Initialize the neural network
    sizes = [784, 128, 2,  10]  # 1# neurons in the input layer, 2# neurons in the hidden layers,
                                # 3# neurons hidden layer, 4# neurons in the output layer
    net = NeuralNetwork(sizes, cost='cross_entropy')

    # Train the neural network
    epochs = 30
    mini_batch_size = 10
    eta = 0.1
    net.fit(X_train, y_train, epochs=epochs, mini_batch_size=mini_batch_size, eta=eta)

    # Evaluate the performance
    accuracy = net.score(X_test, y_test)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")


######################## Second DataSet ########################
def load_second_dataset():
    """
    Load and preprocess the second dataset.

    Returns:
        tuple: Two elements - reduced features (X_reduced) and labels (y).
    """
    data = pd.read_csv('MB_data_train.csv')
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    # Update labels
    y = np.array([1 if str(label).startswith("Pt_Fibro") else 0 for label in y])

    # Normalize the input data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reduce the size of the vectors
    n_components = 10
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced, y


def vectorized_mb_result(j):
    """
    Convert a label into a one-hot encoded vector for the second dataset.

    Args:
        j (int): The label to be converted.

    Returns:
        numpy.ndarray: A 2x1 one-hot encoded vector.
    """
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e


def main_second_dataset(tester=None):
    """
    The main function that loads the second dataset, performs k-fold cross-validation,
    trains a neural network, and evaluates its performance.
    """
    # Load the dataset
    data = pd.read_csv('MB_data_train.csv')

    # Separate features and target
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    y = np.array([0 if str(label).startswith("Pt_Fibro") else 1 for label in y])

    # Preprocess the data by standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define the k-fold cross-validator
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize the neural network parameters
    input_neurons = X.shape[1]
    hidden_neurons = 16
    num_hidden_layers = 1
    output_neurons = 2

    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = [x.reshape(-1, 1) for x in X_train]
        X_test = [x.reshape(-1, 1) for x in X_test]
        y_train = [vectorized_mb_result(y) for y in y_train]
        y_test = np.array(y_test)

        nn = NeuralNetwork([input_neurons, hidden_neurons, num_hidden_layers, output_neurons], cost='cross_entropy')
        nn.fit(X_train, y_train, epochs=30, mini_batch_size=10, eta=0.05)

        accuracy = nn.score(X_test, y_test)
        scores.append(accuracy)
        if not tester:
            print(f"Accuracy on test data: {accuracy * 100:.2f}%")

    if not tester:
        print(f"Average score across {k} folds: {np.mean(scores)}")
    return nn


def main_for_the_tester():
    # Load the dataset
    data = pd.read_csv('tester.csv') # insert here dataset path <-------------------------------------

    # Separate features and target
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    y = np.array([0 if str(label).startswith("Pt_Fibro") else 1 for label in y])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape and convert to appropriate structure
    X = [x.reshape(-1, 1) for x in X]

    net = main_second_dataset(tester=True)

    # Evaluate the performance
    accuracy = net.score(X, y)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main_first_dataset()
    main_second_dataset()  # finds the optimum parameters for the test
    main_for_the_tester()  # insert here the test file


