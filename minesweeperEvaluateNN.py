import torch
from torch.utils.data import DataLoader
from minesweeperTrainNN import MinesweeperDataset, MinesweeperMLP

def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1)  # Sets batch size to 1 for evaluation
    model.eval()  # Sets the model to evaluation mode (disables dropout and batch normalization)

    total_samples = len(dataset)
    correct_predictions = 0

    for board, actual_move in dataloader:
        with torch.no_grad():  # Disables gradient computation during evaluation
            predicted_move = model(board)  # Forwards pass to get predicted move

        # Rounds the predicted move to integers
        predicted_move = torch.round(predicted_move).long()

        # Checks if the predicted move matches the actual move
        if predicted_move[0][0] == actual_move[0][0] and predicted_move[0][1] == actual_move[0][1]:
            correct_predictions += 1

    accuracy = (correct_predictions / total_samples) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    board_size = 10
    # Loads the trained model
    model = MinesweeperMLP(input_size=100, hidden_size=200, output_size=2)
    model.load_state_dict(torch.load('minesweeper_nn.pth'))

    # Loads the new evaluation dataset
    evaluation_dataset = MinesweeperDataset('evaluationData.txt', board_size)

    # Evaluates the model
    evaluate_model(model, evaluation_dataset)

