from tensorflow import *
from keras import *
from keras.layers import *
from neuro import *


def create(dimension):
    model = Sequential([
        Dense(32, input_shape=(pow(dimension, 2),), activation='relu'),
        Dense(64, activation='relu'),
        Dense(256, activation='relu'),
        Dense(pow(dimension, 2), activation='relu', kernel_regularizer='l2')
    ])
    model.compile(optimizer='Adam', loss='mse', metrics='mae')
    return model


def training(model, dimension, package_size, package_number, epochs_number):

    for _ in range(package_number):
        conditions, solutions = generate_examples(dimension, package_size)
        x = conditions_transform(conditions, dimension)
        y = solutions_transform(solutions)
        model.fit(x, y, epochs=epochs_number)


def accuracy_estimation(model, dimension, examples_number):

    conditions, solutions = generate_examples(dimension, examples_number)
    conditions_copy = conditions.copy()

    conditions = conditions_transform(conditions, dimension)
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


def conditions_transform(conditions, dimension):
    return [[condition[j][k] for j in range(dimension) for k in range(dimension)]
            for condition in conditions]


def solutions_transform(solutions):
    return [[solution[i][j] for i in range(len(solution)) for j in range(len(solution))]
            for solution in solutions]


def results_transform(results, dimension):
    return [[[result[i * dimension + j] for j in range(dimension)] for i in range(dimension)]
            for result in results]


