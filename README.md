# SmallEmotionBot
A small (and mostly untrained) text-based emotion analysis bot. Uses a linear regression model with a logistic activation function and 11 Individual Neurons to attempt to capture 11 different emotions. This model uses very high dimensions to accomplish this. In future, I would reduce the dimensionality of weight vectors

# NOTE:
This is currently set up to run on a Windows Device, check comments at the bottom of "cpython.c" and top of "cpython_calculations.py" files to modify for macOS. The code used to train the tokenizer for the model, as well as the tokenizer look-up table, are also present. There is also an empty "token_data.txt" file, which stores the embedded tokens
