from tensorflow import *
from keras import *
from keras.layers import *
import numpy as np
from neuro import *


def create(dimension):
    
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(dimension, dimension, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(pow(dimension, 2), activation='softmax'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    return model


def training(model, dimension, package_size, package_number, epochs_number):

    for _ in range(package_number):
        conditions, solutions = generate_examples(dimension, package_size)
        x = conditions_transform(conditions)
        y = solutions_transform(solutions)
        model.fit(x, y, epochs=epochs_number)


def accuracy_estimation(model, dimension, examples_number):

    conditions, solutions = generate_examples(dimension, examples_number)
    conditions_copy = [[[j for j in i] for i in condition] for condition in conditions]

    conditions = conditions_transform(conditions)
    results = model.predict(conditions)
    results = results_transform(results, dimension)

    results = [solution_transform(result) for result in results]
    results = [collision_avoidance(condition, result) for condition, result in zip(conditions_copy, results)]
    correct_solutions = [1 if result == solution else 0 for result, solution in zip(results, solutions)]
    acccuracy = [objective_function(condition, result) / objective_function(condition, solution)
                 for condition, result, solution in zip(conditions_copy, results, solutions)]
    correct_solutions_proportion = mean(correct_solutions)
    average_accuracy = mean(acccuracy)

    return correct_solutions_proportion, average_accuracy


def conditions_transform(conditions):
    return np.array(conditions)


def solutions_transform(solutions):
    return np.array([[solution[i][j] for i in range(len(solution)) for j in range(len(solution))]
                     for solution in solutions])


def results_transform(results, dimension):
    return [[[result[i * dimension + j] for j in range(dimension)] for i in range(dimension)]
            for result in results]


