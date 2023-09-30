import os
import warnings
import numpy as np
from multiprocessing import Process, Manager
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

# Suppress GPU-related warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras.src.engine.training', message='.*HDF5 file via.*')


# Constants
DATA_CHUNK = 500
NUM_RESULTS = 20


def get_byte_model():
    """
    Builds and returns the byte model.

    Returns:
        Sequential: The byte model structure.
    """
    model = Sequential([
        Dense(32, input_dim=16, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        BatchNormalization(),
        Dense(16, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        BatchNormalization(),
        Dense(2, activation='softmax', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
    ])
    return model


def get_bit_model():
    """
    Builds and returns the bit model.

    Returns:
        Sequential: The bit model structure.
    """
    model = Sequential([
        Dense(8, input_dim=128, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        BatchNormalization(),
        Dense(4, activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        BatchNormalization(),
        Dense(2, activation='softmax', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
    ])
    return model


def training(results, X_train, X_test, y_train, y_test, fold):
    """
    Train a given model and compute its accuracy on test data.

    Args:
    - results (list): List for collecting accuracy results.
    - X_train, X_test, y_train, y_test (numpy arrays): Training and test datasets.
    - fold (int): Current fold number.

    Output:
    - Appends the accuracy of the current fold to the results list.
    """
    model = get_byte_model()  # Can be replaced with get_bit_model()
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model_file_path = os.path.join("models", f"best_model_{fold}.h5")
    modelcheckpoint = ModelCheckpoint(filepath=model_file_path, verbose=0, save_best_only=True)
    earlystopping = EarlyStopping(min_delta=1e-3, patience=5, verbose=0)

    model.fit(X_train, to_categorical(y_train), epochs=1000, batch_size=100, verbose=0,
              validation_split=0.2, callbacks=[modelcheckpoint, earlystopping])

    model = load_model(model_file_path)
    _, acc = model.evaluate(X_test, to_categorical(y_test), verbose=0)
    results.append((fold, acc))


def main():
    """
    Main function to run the training process.
    """
    # Check and create necessary directories
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("logs"):
        os.makedirs("logs")

    X = np.load("X.npy")
    y = np.load("y.npy")
    X, y = shuffle(X, y)

    for data_add in range(NUM_RESULTS):
        train_X, train_y = X[:DATA_CHUNK * (data_add + 1)], y[:DATA_CHUNK * (data_add + 1)]
        print(f"Training with dataset size: {len(train_X)}")

        skf = StratifiedKFold(n_splits=50)

        with Manager() as manager:
            results = manager.list()
            processes = []

            for fold, (train_index, test_index) in enumerate(skf.split(train_X, train_y)):
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
                p = Process(target=training, args=(results, X_train, X_test, y_train, y_test, fold))
                processes.append(p)
                p.start()

            for t in processes:
                t.join()

            result_array = np.zeros((50, 1))
            for item in results:
                result_array[item[0]] = item[1]

        log_file_path = os.path.join("logs", f"results_{data_add}.npy")
        np.save(log_file_path, result_array)
        print(f"Mean Accuracy: {np.mean(result_array):.4f}")


if __name__ == "__main__":
    main()
