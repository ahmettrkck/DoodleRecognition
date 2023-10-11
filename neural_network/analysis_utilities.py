import numpy 

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F

##################################################################################


def show_prediction(image, predicted_prob, labels):
    """
   View image and its predicted category.
    
    ARGS:
        image - image file - tensor
        predicted_prob - predicted probability for each category - tensor
        labels - list of labels
    RETURNS:
        none
    """
    predicted_prob = predicted_prob.data.numpy().squeeze()
    num_categories = len(labels)

    fig, (axis_1, axis_2) = plt.subplots(figsize=(6, 9), ncols=2)
    axis_1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    axis_1.axis('off')
    axis_2.barh(numpy.arange(num_categories), predicted_prob)
    axis_2.set_aspect(0.1)
    axis_2.set_yticks(numpy.arange(num_categories))
    axis_2.set_yticklabels(labels, size='small')
    axis_2.set_title('Category Probability')
    axis_2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


def category_predictions(neural_network, input_vector):
    """
    Returns predicted probabilities from model for each category.

    ARGS:
        neural_network
        input_vector - tensor

    RETURNS:
        prediction vector
    """

    
    with torch.no_grad(): # Speed process by turning off gradient
        logits = neural_network.forward(input_vector)

    return F.softmax(logits, dim=1)


def predicted_labels(prediction_tensor):
    """
        Returns vector of predicted labels for images in dataset.

        ARGS:
            prediction_tensor
        RETURNS:
            label_prediction - (numpy) array of predicted categories for each vector
    """

    np_prediction = prediction_tensor.numpy()
    val_prediction = numpy.amax(np_prediction, axis=1, keepdims=True)
    label_prediction = numpy.array([numpy.where(np_prediction[i, :] == val_prediction[i, :])[
                           0] for i in range(np_prediction.shape[0])])
    label_prediction = label_prediction.reshape(len(np_prediction), 1)

    return label_prediction


def model_evaluation(neural_network, train, y_train, test, y_test):
    """
    Evaluates train and test accuracy of model.

    ARGS:
        neural_network
        train - train dataset - tensor
        y_train - labels for train dataset
        test -  test dataset - tensor
        y_test - labels for test dataset

    RETURNS:
        train_accuracy - train dataset accuracy
        test_accuracy - testing dataset accuracy
    """
    train_prediction = category_predictions(neural_network, train)
    train_prediction_labels = predicted_labels(train_prediction)

    test_prediction = category_predictions(neural_network, test)
    test_prediction_labels = predicted_labels(test_prediction)

    train_accuracy = accuracy_score(y_train, train_prediction_labels)
    test_accuracy = accuracy_score(y_test, test_prediction_labels)

    print("Training dataset accuracy score: {} \n".format(train_accuracy))
    print("Testing dataset accuracy score: {} \n".format(test_accuracy))

    return train_accuracy, test_accuracy
