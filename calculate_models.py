import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import CSVLogger


def nn_model(input_dim, output_dim, nodes=40, dropout_rate=None):
    """Create neural network model with two hidden layers"""
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    if dropout_rate: model.add(Dropout(dropout_rate))
    model.add(Dense(nodes, activation='relu'))
    if dropout_rate: model.add(Dropout(dropout_rate))

    if output_dim == 1:
        model.add(Dense(output_dim, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    data, y = utils.load_data()

    # One hot encoding of target vector
    Y = pd.get_dummies(y).values
    n_classes = Y.shape[1]

    for nodes in [50, 100, 150, 200, 250, 300]:
        for key, X in data.items():
            print('Running : ', key, nodes, X.shape)

            # Split data into training and testing data
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=1, stratify=y)
            
            # Normalize data with mean 0 and std 1
            X_scaled = normalize(X_train)

            # Add callback that streams epoch results to a csv file
            # https://keras.io/callbacks/
            csv_logger = CSVLogger('models/training_{}_{}.log'.format(
                key, nodes))

            # Train the neural network model
            n_features = X.shape[1]
            model = nn_model(n_features, n_classes, nodes, 0.2)
            history = model.fit(X_scaled, Y_train,
                                epochs=100,
                                batch_size=5,
                                verbose=1,
                                callbacks=[csv_logger])

            # Serialize model to JSON
            model_json = model.to_json()
            with open('models/model_{}_{}.json'.format(
                key, nodes), 'w') as f:
                f.write(model_json)

            # Serialize weights to HDF5
            model.save_weights('models/model_{}_{}.h5'.format(
                key, nodes))
