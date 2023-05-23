from tensorflow import *
from keras.layers import Dense
from keras.models import Sequential
from statistics import *
from solve import *


def create_models(size):
    models = [Sequential([
        Dense(2 * pow(size, 2), input_shape=(pow(size, 2),), activation='relu'),
        Dense(4 * pow(size, 2), activation='relu'),
        Dense(16 * pow(size, 2), activation='relu'),
        Dense(size, activation='relu', kernel_regularizer='l2')
    ]) for _ in range(size)]
    [model.compile(optimizer='Adam', loss='mse', metrics='mae') for model in models]
    return models


def models_summary(models):
    [model.summary() for model in models]


def training(models, examples_number, epochs_number):
    size = len(models)

    expenses = []
    solutions = []
    i = 0
    while i < examples_number:
        expense = random_expenses(size)
        try:
            solution = hungarian(expense)
        except:
            solution = None
        if solution is not None:
            expenses.append(expense)
            solutions.append(solution)
            i += 1

    x = [[expenses[i][j][k] for j in range(size) for k in range(size)]
         for i in range(examples_number)]
    y = [[solutions[j][i] for j in range(examples_number)] for i in range(size)]
    [model.fit(x, example, epochs=epochs_number) for model, example in zip(models, y)]


def models_save(models, name=''):
    [keras.models.save_model(models[i], name + str(i)) for i in range(len(models))]


def models_load(size, name=''):
    return [keras.models.load_model(name + str(i)) for i in range(size)]


def accuracy_estimation(models, expenses_size, number_of_test_cases):
    models_size = len(models)
    if models_size < expenses_size:
        print('error')
        return

    neuro_correct_decision = []
    neuro_acccuracy = []
    random_correct_decision = []
    random_accuracy = []

    for _ in range(number_of_test_cases):

        expenses = random_expenses(expenses_size)

        solutions = None
        while solutions is None:
            try:
                solutions = hungarian(expenses)
            except:
                solutions = None

        neuro_expenses = expenses.copy()

        if models_size > expenses_size:
            neuro_expenses = extension(neuro_expenses, models_size - expenses_size)

        neuro_expenses = [[neuro_expenses[i][j] for i in range(models_size) for j in range(models_size)]]
        neuro_solutions = [list(model.predict(neuro_expenses)[0]) for model in models]

        if models_size > expenses_size:
            neuro_solutions = contraction(neuro_solutions, models_size - expenses_size)

        neuro_solutions = [solution_transformation(solution) for solution in neuro_solutions]
        neuro_solutions = collision_avoidance(expenses, neuro_solutions)

        neuro_correct_decision.append(1 if solutions == neuro_solutions else 0)
        neuro_acccuracy.append(objective_function(expenses, neuro_solutions) / objective_function(expenses, solutions))

        random_solutions = random_solution(expenses_size)

        random_correct_decision.append(1 if solutions == random_solutions else 0)
        random_accuracy.append(objective_function(expenses, random_solutions) / objective_function(expenses, solutions))

    neuro_proportion_correct_decision = mean(neuro_correct_decision)
    neuro_average_accuracy = mean(neuro_acccuracy)
    random_proportion_correct_decision = mean(random_correct_decision)
    random_average_accuracy = mean(random_accuracy)

    ratio_proportion_correct_decision = neuro_proportion_correct_decision / random_proportion_correct_decision
    ratio_average_accuracy = neuro_average_accuracy / random_average_accuracy

    print('neuro:    ', neuro_proportion_correct_decision, neuro_average_accuracy)
    print('random:   ', random_proportion_correct_decision, random_average_accuracy)
    print(ratio_proportion_correct_decision, ratio_average_accuracy)

