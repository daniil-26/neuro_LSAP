from neuro import *
import fnn
import cnn


if __name__ == '__main__':

    dimension = 4
    package_size = 1000
    package_number = 50
    epochs_number = 30
    examples_number = 10000

    model = cnn.create(dimension=dimension)
    data(model)
    cnn.training(model=model,
                 dimension=dimension,
                 package_size=package_size,
                 package_number=package_number,
                 epochs_number=epochs_number)
    cnn_accuracy = cnn.accuracy_estimation(model=model,
                                           dimension=dimension,
                                           examples_number=examples_number)

    model = fnn.create(dimension=dimension)
    data(model)
    fnn.training(model=model,
                 dimension=dimension,
                 package_size=package_size,
                 package_number=package_number,
                 epochs_number=epochs_number)
    fnn_accuracy = fnn.accuracy_estimation(model=model,
                                           dimension=dimension,
                                           examples_number=examples_number)

    greedy_accuracy = accuracy_estimation(method=greedy,
                                          dimension=dimension,
                                          examples_number=examples_number)

    random_accuracy = accuracy_estimation(method=random_method,
                                          dimension=dimension,
                                          examples_number=examples_number)

    print('cnn:        ', *cnn_accuracy)
    print('fnn:        ', *fnn_accuracy)
    print('greedy:     ', *greedy_accuracy)
    print('random:     ', *random_accuracy)


