# Polygon_RNN
In this code, we define the PolygonRNN class that inherits from tf.keras.Model. It uses an embedding layer to map the input sequence (represented as integers) to a continuous space, then applies a GRU layer to capture the sequential dependencies, and finally a dense layer to predict the (x, y) coordinates of the polygon vertices. The generate_data function is used to create random training data with the specified number of sides.

We set the hyperparameters such as the number of sides, hidden size, batch size, and number of epochs. Then, we create an instance of the model, compile it with the Adam optimizer and mean squared error loss function, generate training data, and train the model using the fit method.

Finally, we generate a polygon using the trained model by providing an example input (X_test) and calling the predict method. The predicted (x, y) coordinates of the polygon vertices are printed to the console.

Note that this is a simple example to demonstrate the structure of a Polygon RNN. Depending on your specific requirements, you may need to modify or extend the code.