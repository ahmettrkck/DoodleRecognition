import numpy # Provides advanced mathematical operations

import requests # Used for making HTTP requests
from io import BytesIO  # Used for allowing file related inputs
import pickle # Used for converting object into byte stream to be stored in pickle file
from collections import OrderedDict # Provides contained including lists, tuples, dictionaries and arrays
from os import path # Used for fetching and changing contents of directory
import argparse # Used for allowing positional arguments


from sklearn.model_selection import train_test_split # Splits arrays into random train and test subsets

import torch # Deep learning tensor library

import constants # Allows to use constants for no. of epochs, mini-batches and categories

import sys
sys.path.insert(0, '../modules')
from image_utilities import enrich_images

###############################################################################################################

class ModelBuilder:

    def __init__(self, data_dir, model_input_size, model_hidden_sizes, model_output_size, dropout=0.0):
        '''
        Feed-forward neural network with one hidden layer is created using passed parameters.
        
        ARGS:
            data_dir - data directory
            model_input_size - size of the input layer
            model_output_size - list of the hidden layer sizes
            model_hidden_sizes
            dropout - probability of keeping node

        RETURNS: -
        '''
        self.data_dir = data_dir
        self.model_input_size = model_input_size
        self.model_hidden_sizes = model_hidden_sizes
        self.model_output_size = model_output_size
        self.dropout = dropout
    

    def load_dataset(self, labels, enrich_data):
        """

        OBJECTIVE: Dataset is either loaded from the web or from the pickle files. Dataset is enriched if enrich_data=true.

        ARGS:
            labels - list of labels
            enrich_data - boolean, dataset is enriched if true

        RETURNS: -

        """
        print("Loading dataset \n")

        # Check if data has been loaded to pickle files
        if not(path.exists('training_data/doodle_xtrain.pickle')):
            
            print("Gathering data from Google \n")

            # Dictionary containing URLs for the selected categories
            url_dict = {}
            for cat in labels:
                url_dict[cat] = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + cat + '.npy'

            # Data is loaded for each category into the dictionary
            dict_categories = {}
            for key, url in url_dict.items():
                response = requests.get(url)
                dict_categories[key] = numpy.load(BytesIO(response.content))
            print("Data is loaded \n")

            # Labels are generated and added to the loaded data
            for i, (key, value) in enumerate(dict_categories.items()):
                value = value.astype('float32')/255.
                if i == 0:
                    dict_categories[key] = numpy.c_[value, numpy.zeros(len(value))]
                else:
                    dict_categories[key] = numpy.c_[value, i*numpy.ones(len(value))]

            doodle_list = []
            for key, value in dict_categories.items():
                doodle_list.append(value[:5000])
            doodles = numpy.concatenate(doodle_list)

            # Data is split into features (X) and category labels (y)
            y = doodles[:, -1].astype('float32')
            X = doodles[:, : constants.INPUT_SIZE]

            # Dataset is split into train and test data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=1)
        else:
            # Data loaded from pickle files
            print("Data already loaded \n")

            pickle_file = open(self.data_dir + '/' + "doodle_xtrain.pickle", 'rb')
            self.X_train = pickle.load(pickle_file) # training dataset
            pickle_file.close()

            pickle_file = open(self.data_dir + '/' + "doodle_xtest.pickle", 'rb')
            self.X_test = pickle.load(pickle_file) # testing dataset
            pickle_file.close()

            pickle_file = open(self.data_dir + '/' + "doodle_ytrain.pickle", 'rb')
            self.y_train = pickle.load(pickle_file) # training dataset labels
            pickle_file.close()

            pickle_file = open(self.data_dir + '/' + "doodle_ytest.pickle", 'rb')
            self.y_test = pickle.load(pickle_file) # testing dataset labels
            pickle_file.close()

        # Flipped and rotated images are added to the dataset
        if (enrich_data):
            print("Enriching dataset \n")
            self.X_train, self.y_train = enrich_images(self.X_train, self.y_train, constants.INPUT_SIZE)


    def save_training_data(self, save_always=False):
        """
        Dataset is saved to pickle files.

        ARGS:
            save_always - (boolean) file is saved automatically

        RETURNS: -
        """

        # Check if files have been saved
        if save_always or not(path.exists(self.data_dir + '/' + 'doodle_xtrain.pickle')):
            
            with open(self.data_dir + '/' + 'doodle_xtrain.pickle', 'wb') as file: # Save X_train dataset
                pickle.dump(self.X_train, file)

        
            with open(self.data_dir + '/' + 'doodle_xtest.pickle', 'wb') as file:  # Save X_test dataset
                pickle.dump(self.X_test, file)

        
            with open(self.data_dir + '/' + 'doodle_ytrain.pickle', 'wb') as file:  # Save y_train dataset
                pickle.dump(self.y_train, file)

        
            with open(self.data_dir + '/' + 'doodle_ytest.pickle', 'wb') as file:   # Save y_test dataset
                pickle.dump(self.y_test, file)
    

    def create_model(self):
        '''
        Feed-forward neural network with one hidden layer is created using passed parameters.
        
        ARGS:
            model_input_size - size of the input layer
            model_output_size - list of the hidden layer sizes
            model_hidden_sizes
            dropout - probability of keeping node

        RETURNS: -
        '''

        self.neural_network = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(self.model_input_size, self.model_hidden_sizes[0])),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(
                self.model_hidden_sizes[0], self.model_hidden_sizes[1])),
            ('bn2', torch.nn.BatchNorm1d(
                num_features=self.model_hidden_sizes[1])),
            ('relu2', torch.nn.ReLU()),
            ('dropout', torch.nn.Dropout(self.dropout)),
            ('fc3', torch.nn.Linear(
                self.model_hidden_sizes[1], self.model_hidden_sizes[2])),
            ('bn3', torch.nn.BatchNorm1d(
                num_features=self.model_hidden_sizes[2])),
            ('relu3', torch.nn.ReLU()),
            ('logits', torch.nn.Linear(self.model_hidden_sizes[2], self.model_output_size))]))
 

    def save_model_to_file(self, filepath='stored_model.pth'):
        """
        Saves model to file.

        ARGS:
            filepath - path for the model to be saved to

        RETURNS: -
        """

        print("Saving model to {}\n".format(filepath))
        stored_model = {'model_input_size': self.model_input_size,
                    'model_output_size':  self.model_output_size,
                    'model_hidden_layers': self.model_hidden_sizes,
                    'dropout': self.dropout,
                    'state_dict': self.neural_network.state_dict()}

        torch.save(stored_model, filepath)


    def train_model(self, epochs=50, n_chunks=1200, learning_rate=0.003, weight_decay=0, optimiser='SGD'):
        """
        Trains the model.

        ARGS:
            epochs - no. epochs
            n_chunks - no. chunks to split dataset
            learning_rate
            weight_decay
            optimiser

        RETURNS: None
        """

        print("Fitting model with epochs = {epochs}, learning rate = {lr}\n"
            .format(epochs=epochs, lr=learning_rate))

        self.convert_np_to_tensor()

        cel_crit = torch.nn.CrossEntropyLoss()

        if (optimiser == 'SGD'):
            optimiser = torch.optim.SGD(self.neural_network.parameters(),
                                        lr=learning_rate, weight_decay=weight_decay)
        else:
            optimiser = torch.optim.Adam(
                self.neural_network.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print_n = 100

        step = 0

        for epoch in range(epochs):
            loss_count = 0

            self.X_train, self.y_train = shuffle_xy_train(self.X_train, self.y_train)

            image_chunks = torch.chunk(self.X_train, n_chunks)
            label_chunks = torch.chunk(self.y_train, n_chunks)

            for i in range(n_chunks):
                step += 1

                optimiser.zero_grad()

                output = self.neural_network.forward(image_chunks[i]) # Forward / backward pass
                loss = cel_crit(output, label_chunks[i].squeeze())
                loss.backward()
                optimiser.step()

                loss_count += loss.item()

                if step % print_n == 0:
                    print("Epoch - {}/{}  ".format(epoch+1, epochs),
                        "Loss - {:.4f}".format(loss_count/print_n))

                    loss_count = 0


    def convert_np_to_tensor(self):
        self.X_train = torch.from_numpy(self.X_train).float()
        self.y_train = torch.from_numpy(self.y_train).long()
        self.X_test = torch.from_numpy(self.X_test).float()
        self.y_test = torch.from_numpy(self.y_test).long()


def shuffle_xy_train(X_train, y_train):
    """
    Training data is shuffled.
    ARGS:
        X_train
        y_train

    RETURNS:
        shuffled_X_train
        shuffled_y_train
    """
    shuffled_X_train = X_train.numpy()
    shuffled_y_train = y_train.numpy().reshape((X_train.shape[0], 1))

    random_order = list(numpy.random.permutation(X_train.shape[0]))
    shuffled_X_train = shuffled_X_train[random_order, :]
    shuffled_y_train = shuffled_y_train[random_order, :].reshape(
        (X_train.shape[0], 1))

    shuffled_X_train = torch.from_numpy(shuffled_X_train).float()
    shuffled_y_train = torch.from_numpy(shuffled_y_train).long()

    return shuffled_X_train, shuffled_y_train  


def main():
    parser = argparse.ArgumentParser(description='Argument parser')

    parser.add_argument('--model_dir', action='store', default='.',
                        help='Directory for produced model')

    parser.add_argument('--data_dir', action='store', default='training_data',
                        help='Directory for training data')

    parser.add_argument('--learning_rate', type=float, action='store', default=0.003,
                        help='Learning rate')

    parser.add_argument('--epochs', type=int, action='store', default=50,
                        help='Epochs')

    parser.add_argument('--weight_decay', type=float, action='store', default=0,
                        help='Weight decay')

    parser.add_argument('--dropout', type=float, action='store', default=0.0,
                        help='Dropout')

    parser.add_argument('--enrich_data', action='store_true', default=False,
                        help='Add rotated and flipped images to training data.')

    parser.add_argument('--chunks', type=int, action='store', default=1200,
                        help='No. of chunks.')

    parser.add_argument('--optimiser', action='store', default='SGD',
                        choices=['SGD', 'Adam'],
                        help='optimiser to fit model.')

    
    parsing_results = parser.parse_args()
    

    print("\nParamaters: {parsing_results}\n".format(parsing_results=parsing_results))

    model_path = parsing_results.model_dir + '/' + 'stored_model.pth'
    data_dir = parsing_results.data_dir
    learning_rate = parsing_results.learning_rate
    epochs = parsing_results.epochs
    weight_decay = parsing_results.weight_decay
    dropout = parsing_results.dropout
    n_chunks = parsing_results.chunks
    optimiser = parsing_results.optimiser

    # Network's hyper-parameters
    model_input_size =  constants.INPUT_SIZE
    model_hidden_sizes = [128, 100, 64]
    model_output_size = constants.NUM_CATEGORIES

    model_builder = ModelBuilder(data_dir, model_input_size, model_hidden_sizes, model_output_size, dropout=dropout)
    
    model_builder.load_dataset(constants.CATEGORY_LABEL_DICT.values(), parsing_results.enrich_data) # Load data and enrich if required

    # New dataset saved to file
    model_builder.save_training_data(save_always=parsing_results.enrich_data)

    model_builder.create_model()   # Model  built
   
    model_builder.train_model(epochs=epochs, n_chunks=n_chunks,
                                learning_rate=learning_rate, weight_decay=weight_decay, optimiser=optimiser)  # Model fitted

    model_builder.save_model_to_file(filepath=model_path)  # Model saved


if __name__ == '__main__':
    main()
