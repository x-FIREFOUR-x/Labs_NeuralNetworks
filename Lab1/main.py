from tensorflow import keras
import numpy as np


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

if __name__ == '__main__':
    model = keras.Sequential()
    model.add(keras.layers.Dense(15, input_dim=2, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x, y, epochs=500, verbose=1)

    scores = model.evaluate(x, y)
    print(np.round(model.predict(x)))
    print(y)


