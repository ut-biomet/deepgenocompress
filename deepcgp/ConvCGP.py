# ==============================
# Tanzila Islam
# Email: tanzilamohita@gmail.com
# Date: 5/27/2024
# ===============================

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


def create_model(input_shape):
    # there are 13 layers defined in this model  except input layer
    model = Sequential()
    model.add(Conv1D(32, 3, activation=None, kernel_initializer='he_normal', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, 7, activation=None, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(96, 5, activation=None, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256, 5, activation=None, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(256, activation=None, kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(224, activation=None, kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.0096039)
    model.compile(loss='mse', optimizer=optimizer)

    model.summary()
    return model
