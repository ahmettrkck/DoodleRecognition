import numpy
from PIL import Image
import base64
from io import BytesIO
import base64
import time
from collections import OrderedDict
import json

import matplotlib.pyplot as plot

import plotly
import plotly.graph_objs as go

from flask import Flask
from flask import render_template, request

# Import image processing utilities
import sys
sys.path.insert(0, '../modules')
from image_utilities import crop, normalise_image, rgba_to_rgb, convert_PIL_to_np

import torch
from torch import nn
import torch.nn.functional as F

import constants

def load_nn(nn_path='stored_model.pth'):
    """
    Neural network model is loaded from file.

    ARGS:
        nn_path - path for saved neural network

    RETURNS:
        neural_network - loaded neural network
    """

    print("Loading model: {} \n".format(nn_path))

    stored_model = torch.load(nn_path)
    model_input_size = stored_model['model_input_size']
    model_output_size = stored_model['model_output_size']
    model_hidden_sizes = stored_model['model_hidden_layers']
    dropout = stored_model['dropout']
    neural_network = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model_input_size, model_hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(model_hidden_sizes[0], model_hidden_sizes[1])),
                          ('bn2', nn.BatchNorm1d(
                              num_features=model_hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(model_hidden_sizes[1], model_hidden_sizes[2])),
                          ('bn3', nn.BatchNorm1d(
                              num_features=model_hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('logits', nn.Linear(model_hidden_sizes[2], model_output_size))]))
    neural_network.load_state_dict(stored_model['state_dict'])

    return neural_network


def get_all_preds(model, input):
    """
    
    Probabilities of the selected categories are returned and the category with the greatest is also returned.
    
    ARGS:
        model
        input - vector

    RETURNS:
        predictions
        predicted_label - predicted category's label name
        
    """
   
    input = torch.from_numpy(input).float()  # Input converted to tensor
    input = input.resize_(1, constants.INPUT_SIZE)

    
    with torch.no_grad():
        logits = model.forward(input) # Increase efficiency by turning off gradient
    ps = F.softmax(logits, dim=1)

    predictions = ps.numpy()

   
    category_key = numpy.argmax(predictions)  # Get key with greatest prob
    predicted_label = constants.CATEGORY_LABEL_DICT[category_key]  # Get category name from dictionary

    return predictions, predicted_label 


def save_prediction_image(image, predictions, dir='user_doodles'):
    """
    
    Prediction image and predicted category are saved.
    
    ARGS:
        image - image file
        preds - (numpy) predicted probabilities for each category
    """
    predictions = predictions.squeeze()

    plot.switch_backend('Agg')  # Required to run on Mac
    fig, (axis_1, axis_2) = plot.subplots(figsize=(6, 9), ncols=2)

    num_categories = len(constants.CATEGORY_LABEL_DICT)
    labels = constants.CATEGORY_LABEL_DICT.values()

    axis_1.barh(numpy.arange(num_categories), predictions)
    axis_1.set_aspect(0.1)
    axis_1.set_yticks(numpy.arange(num_categories))
    axis_1.set_yticklabels(labels, size='small')
    axis_1.set_title('Probability of categories')
    axis_1.set_xlim(0, 1.1)

    axis_2.imshow(image.squeeze())
    axis_2.axis('off')

    plot.tight_layout()

    timestamp = time.time()
    plot.savefig(dir + '/' + 'prediction' + str(timestamp) + '.png')


app = Flask(__name__)


model = load_nn() # load model
model.eval() 

# web app endpoints

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def show_pred():
    """
    Display prediction
    """
    # Get image data in URL parameter, and decode to base64 string
    base64_image = request.args.get('doodle').replace('.', '+').replace('_', '/').replace('-', '=')

    # String is converted to bytes
    byte_data = base64.b64decode(base64_image)
    image_data = BytesIO(byte_data)

   
    image = Image.open(image_data).convert("RGBA")

    # image is processed for the neural network model

    image_cropped = crop(image)

    image_normalised = normalise_image(image_cropped) 

    image_rgb = rgba_to_rgb(image_normalised)

    image_np = convert_PIL_to_np(image_rgb) # preprocessed image in numpy format

    # pass image to model and print category predictions received
    predictions, predicted_label  = get_all_preds(model, image_np)
    print("This is a {}".format(predicted_label))

    save_prediction_image(image_np, predictions)

    # graph of category probabilities is created
    category_plot = [
        {
            'data': [
                go.Bar(
                    x=predictions.ravel().tolist(),
                    y=list(constants.CATEGORY_LABEL_DICT.values()),
                    orientation='h')
            ],

            'layout': {
                'title': 'Category Probabilities',
                'yaxis': {
                    'title': "Categories"
                },
                'xaxis': {
                    'title': "Probability",
                }
            }
        }]

    # graphs are encoded in JSON to be passed to the front end
    graphIDs = ["graph-{}".format(i) for i, _ in enumerate(category_plot)]
    graph_data = json.dumps(category_plot, cls=plotly.utils.PlotlyJSONEncoder)

    # graph is rendered onto front end using the Flask template
    return render_template(
        'result.html',
        result=predicted_label,  # predicted category label
        graph_ids=graphIDs,  
        graphJSON=graph_data,  
        imageString=base64_image  # image displayed in base64
    )


def main():
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    main()
