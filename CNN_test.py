import numpy as np
import torch
from CNN import one_hot_encode, MinesweeperCNN

def load_trained_model(model_class, board_size, model_filename):
    model = model_class(board_size)  # Initialize the model class
    model.load_state_dict(torch.load(model_filename))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    return model

def prepare_board(board, size_x, size_y):
    # Convert the board to the one-hot encoded format
    board_numerical = one_hot_encode(board, size_x, size_y)
    board_tensor = torch.tensor(board_numerical, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return board_tensor

def predict_risk(model, board_tensor):
    with torch.no_grad():  # Disable gradient computation
        output = model(board_tensor)  # Get model output
    return output

def process_predictions(predictions, size_x, size_y):
    predictions = predictions.squeeze().numpy()  # Remove batch dimension and convert to numpy array
    return predictions.reshape(size_x, size_y)  # Reshape to board dimensions

def board_to_string(board, unknown_value=9.0, artificial_value=9.0):
    # Convert each cell to float, handling unknown and artificial values
    flattened_board = [
        unknown_value if cell == '?' or cell == ' ' else
        artificial_value if cell == '_' else
        float(cell)
        for row in board for cell in row
    ]
    # Convert the list into a numpy ndarray with shape (25,)
    board_ndarray = np.array(flattened_board, dtype=float)
    return board_ndarray

def board_to_string1(board, unknown_value=9.0, artificial_value=9.0):
    # Flatten the 2D board list into a 1D list of floats
    flattened_board = [
        unknown_value if cell == '?' or cell == ' ' else
        artificial_value if cell == '_' else
        float(cell)  # Convert numeric characters to float
        for row in board for cell in row
    ]
    # Convert the list into a numpy ndarray with shape (25,)
    board_ndarray = np.array(flattened_board, dtype=float)
    return board_ndarray


if __name__ == '__main__':
    SIZE = (5, 5)  # Board size
    MODEL_FILENAME = 'MODELS/Model_(5, 5).pth'  # Path to your trained model file

    # Initialize the model and load the trained weights
    model = load_trained_model(MinesweeperCNN, SIZE, MODEL_FILENAME)
    '''
        2 1 0 0 0
        ? 2 1 1 0
        1 3 ? 2 0
        0 2 ? 2 0
        0 1 1 1 0
    '''
    # Example board (replace with your actual example)
    example_board = [
        ['_', '_', '_', '_', '_'],
        ['_', '1.0', '1.0', '1.0', '0.0'],
        ['_', '2.0', '?', '1.0', '0.0'],
        ['_', '2.0', '2.0', '2.0', '0.0'],
        ['_', '1.0', '0.0', '1.0', '0.0']
    ]

    # Prepare the board for prediction

    board_string = board_to_string(example_board)

    # Print the resulting string
    print(board_string)

    board_tensor = prepare_board(board_string, *SIZE)

    # Get predictions from the model
    predictions = predict_risk(model, board_tensor)

    # Process and print the predictions
    #risk_board = process_predictions(predictions, *SIZE)
    print("Predicted Risk Board:")
    print(predictions)
