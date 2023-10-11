# Overview of Doodle Recogniser
This is a neural network model that detects doodles drawn on a canvas by the user.
It gathers the selected categories from Google’s Quick-Draw dataset API, and stores these images
within files in order to avoid having to download from the web again.
These images are pre-processed in order to create a greater range of data
to be fed into the neural network for testing and training purposes. Once the model is trained,
it is stored in a file, so that it can be used readily offline.

To add functionality, a web application with a friendly-user interface allows the user
to draw on a canvas, clear the canvas and submit their drawing to the neural network
for identification. In turn, the network outputs the corresponding probability that
this drawing belongs to each category, and selects the highest probability as the final recognition.

The user is able to submit another drawing until they close the application.
All drawn images are stored in order for them to be later fed back into the training data
to create an even stronger model. To help the evaluation of the success of the model,
and for optimisation purposes, there are several utility programs.

# Instructions
## Running model_builder
```
cd neural_network
mkdir -p training_data
python model_builder.py --data_dir=./training_data --model_dir=../doodle_recogniser --epochs=50 --chunks=1200

# Or use enrich_data flag
python model_builder.py --data_dir=./training_data --model_dir=../doodle_recogniser --enrich_data --epochs=50 --chunks=1200
```

## Running app_runner
```
cd doodle_recogniser
mkdir -p user_doodles
python app_runner.py

```