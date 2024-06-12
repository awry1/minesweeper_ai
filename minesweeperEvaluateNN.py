import torch
from torch.utils.data import DataLoader
from minesweeperTrainNN import MinesweeperDataset, MinesweeperMLP  # Import your model and dataset class

def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1)  # Set batch size to 1 for evaluation
    model.eval()  # Set the model to evaluation mode (disables dropout and batch normalization)

    total_samples = len(dataset)
    correct_predictions = 0

    for board, actual_move in dataloader:
        with torch.no_grad():  # Disable gradient computation during evaluation
            predicted_move = model(board)  # Forward pass to get predicted move

        # Round the predicted move to integers
        predicted_move = torch.round(predicted_move).long()

        # Check if the predicted move matches the actual move
        if predicted_move[0][0] == actual_move[0][0] and predicted_move[0][1] == actual_move[0][1]:
            correct_predictions += 1

    accuracy = (correct_predictions / total_samples) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    board_size = 10  # Change this to the size of your boards
    # Load the trained model
    model = MinesweeperMLP(input_size=100, hidden_size=200, output_size=2)  # Modify input_size and hidden_size accordingly
    model.load_state_dict(torch.load('minesweeper_nn.pth'))  # Load the trained weights

    # Load the new evaluation dataset
    evaluation_dataset = MinesweeperDataset('evaluationData.txt', board_size)  # Replace 'evaluationData.txt' with your evaluation dataset file path

    # Evaluate the model
    evaluate_model(model, evaluation_dataset)