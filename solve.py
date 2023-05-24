import random
from hungarian_algorithm import algorithm

random.seed()


def hungarian(e):
    n = len(e)
    g = {str(i) + '_': {str(j): e[i][j] for j in range(n)} for i in range(n)}
    s = algorithm.find_matching(g, matching_type='max', return_type='list')
    s = [x[0] for x in s]
    s = [[1 if (str(i) + '_', str(j)) in s else 0 for j in range(n)] for i in range(n)]
    return s


def greedy(expenses):
    size = len(expenses)
    solution = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        j = expenses[i].index(max(expenses[i]))
        solution[i][j] = 1
        for k in range(i + 1, size):
            expenses[k][j] = -1
    return solution


def random_expenses(size):
    return [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]


def collision_avoidance(expenses, solution):
    n = min(len(expenses), len(solution))
    for i in range(n):
        if [solution[j][i] for j in range(n)].count(1) > 1:
            max_parameter = max([expenses[j][i] for j in range(n) if solution[j][i]])
            for j in range(n):
                if expenses[j][i] < max_parameter:
                    solution[j][i] = 0
    for i in range(n):
        if 1 not in solution[i]:
            max_parameter = max(expenses[i][j] for j in range(n) if 1 not in [solution[k][j] for k in range(n)])
            for j in range(n):
                if expenses[i][j] == max_parameter and 1 not in [solution[k][j] for k in range(n)]:
                    solution[i][j] = 1
                    break
    return solution


def solution_transformation(solution):
    x = solution.index(max(solution))
    return [1 if i == x else 0 for i in range(len(solution))]


def random_solution(size):
    x = list(range(0, size))
    random.shuffle(x)
    return [[1 if j == x[i] else 0 for j in range(size)] for i in range(size)]


def objective_function(expenses, solution):
    n = min(len(expenses), len(solution))
    return sum(sum([[expenses[i][j] * solution[i][j] for j in range(n)] for i in range(n)], []))


def extension(array, x):
    n = len(array)
    array = [a + [0] * x for a in array]
    array += [[0] * (n + x) for _ in range(x)]
    for i in range(n, n + x):
        array[i][i] = 100
    return array


def contraction(array, x):
    return [a[0:-x] for a in array[:-x]]


