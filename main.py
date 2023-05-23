from neuro import *


if __name__ == '__main__':

    models = create_models(size=4)

    training(models=models,
             examples_number=50000,
             epochs_number=30)

    accuracy_estimation(models=models,
                        expenses_size=4,
                        number_of_test_cases=1000)

