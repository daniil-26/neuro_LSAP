from statistics import *
from solve import *


def data(model):
    model.summary()


def generate_examples(dimension, examples_number):

    conditions = []
    solutions = []

    i = 0
    while i < examples_number:
        expense = random_condition(dimension)
        try:
            solution = hungarian(expense)
        except:
            solution = None
        if solution is not None:
            conditions.append(expense)
            solutions.append(solution)
            i += 1

    return conditions, solutions


def collision_avoidance(condition, solution):
    n = len(condition)
    for i in range(n):
        if [solution[j][i] for j in range(n)].count(1) > 1:
            m = max([condition[j][i] for j in range(n) if solution[j][i]])
            for j in range(n):
                if condition[j][i] < m:
                    solution[j][i] = 0
    for i in range(n):
        if 1 not in solution[i]:
            m = max(condition[i][j] for j in range(n) if 1 not in [solution[k][j] for k in range(n)])
            for j in range(n):
                if condition[i][j] == m and 1 not in [solution[k][j] for k in range(n)]:
                    solution[i][j] = 1
                    break
    return solution


def solution_transform(solution):
    dimension = len(solution)
    for i in range(dimension):
        m = solution[i].index(max(solution[i]))
        solution[i] = [1 if j == m else 0 for j in range(dimension)]
    return solution


def accuracy_estimation(method, dimension, examples_number):
    conditions, solutions = generate_examples(dimension, examples_number)

    results = [method(condition) for condition in conditions]

    correct_solutions = [1 if result == solution else 0 for result, solution in zip(results, solutions)]
    acccuracy = [objective_function(condition, result) / objective_function(condition, solution)
                 for condition, result, solution in zip(conditions, results, solutions)]
    correct_solutions_proportion = mean(correct_solutions)
    average_accuracy = mean(acccuracy)

    return correct_solutions_proportion, average_accuracy


