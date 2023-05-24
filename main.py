from neuro import *


if __name__ == '__main__':

    model = create_model(size=4)
    training_model(model=model,
                   size=4,
                   examples_number=50000,
                   epochs_number=30)
    model_accuracy_estimation(model=model,
                              model_size=4,
                              expenses_size=4,
                              number_of_test_cases=1000)
    model_accuracy_estimation(model=model,
                              model_size=4,
                              expenses_size=3,
                              number_of_test_cases=1000)
    model_accuracy_estimation(model=model,
                              model_size=4,
                              expenses_size=2,
                              number_of_test_cases=1000)
    
    models = create_models(size=4)
    training(models=models,
             examples_number=50000,
             epochs_number=30)
    accuracy_estimation(models=models,
                        expenses_size=4,
                        number_of_test_cases=1000)
    accuracy_estimation(models=models,
                        expenses_size=3,
                        number_of_test_cases=1000)
    accuracy_estimation(models=models,
                        expenses_size=3,
                        number_of_test_cases=1000)
