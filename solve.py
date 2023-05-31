import random
from hungarian_algorithm import algorithm

random.seed()


def hungarian(condition):
    dimension = len(condition)
    condition = {str(i) + '_': {str(j): condition[i][j] for j in range(dimension)} for i in range(dimension)}
    solution = algorithm.find_matching(condition, matching_type='max', return_type='list')
    solution = [x[0] for x in solution]
    solution = [[1 if (str(i) + '_', str(j)) in solution else 0 for j in range(dimension)] for i in range(dimension)]
    return solution


def greedy(condition):
    condition_copy = condition.copy()
    dimension = len(condition_copy)
    solution = [[0 for _ in range(dimension)] for _ in range(dimension)]
    for i in range(dimension):
        j = condition_copy[i].index(max(condition_copy[i]))
        solution[i][j] = 1
        for k in range(i + 1, dimension):
            condition_copy[k][j] = -1
    return solution


def random_method(condition):
    dimension = len(condition)
    return random_solution(dimension)


def random_condition(dimension):
    return [[random.random() for _ in range(dimension)] for _ in range(dimension)]


def random_solution(dimension):
    x = list(range(0, dimension))
    random.shuffle(x)
    return [[1 if j == x[i] else 0 for j in range(dimension)] for i in range(dimension)]


def objective_function(condition, solution):
    dimension = len(condition)
    return sum(sum([[condition[i][j] * solution[i][j] for j in range(dimension)] for i in range(dimension)], []))


def extension(array, x):
    n = len(array)
    array = [a + [0] * x for a in array]
    array += [[0] * (n + x) for _ in range(x)]
    for i in range(n, n + x):
        array[i][i] = 100
    return array


def contraction(array, x):
    return [a[0:-x] for a in array[:-x]]


