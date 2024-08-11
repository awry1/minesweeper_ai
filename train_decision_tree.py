import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Constants for quick change
BOARD_SIZE = 10, 10
ITERATIONS = 1000
MINES = 10

# Code to process CNN outputs and windows given
# (Probably have to modify test_results to save windows, not entire boards
# and train decision tree. After that, run a test and visualize the tree itself
# for clear interpretation)


# Step 1: Convert Windows to Numerical Features
def window_to_numeric(window):
    mapping = {'_': -1.0, '0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, 
                   '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, '?': 9.0}

    numeric_window = []
    for row in window:
        numeric_row = [mapping[cell] for cell in row.split()]
        numeric_window.extend(numeric_row)

    return np.array(numeric_window)


# Step 2: Classify the CNN Risk Output
def classify_risk(risk_score):
    if risk_score <= 0.25:
        return 0  # "not a mine"
    elif risk_score <= 0.5:
        return 1  # "probably not a mine"
    elif risk_score <= 0.75:
        return 2  # "probably mine"
    else:
        return 3  # "mine"


# Step 3: Prepare the Dataset
def load_test_result(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Omit the first two lines; not needed here
    lines = lines[2:]

    windows = []
    risks = []

    current_window = []
    current_risk = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == 'Risk:':
            current_risk = float(lines[i + 1].strip())
            i += 2  # Skip the 'Risk:' line and the risk value line
            continue

        if line == '':
            if current_window and current_risk is not None:
                windows.append(current_window)
                risks.append(current_risk)
                current_window = []
                current_risk = None
            i += 1
            continue

        current_window.append(line)
        i += 1

    if current_window and current_risk is not None:
        windows.append(current_window)
        risks.append(current_risk)

    return windows, risks


# Step 4: Prepare the Dataset
def prepare_dataset(windows, risks):
    X = []
    y = []

    for window, risk in zip(windows, risks):
        numeric_window = window_to_numeric(window)
        X.append(numeric_window)
        y.append(classify_risk(risk))

    return np.array(X), np.array(y)


# Step 5: Train the Decision Tree
def train_decision_tree(X, y):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set max_depth to None for a fully grown tree or adjust as needed
    decision_tree = DecisionTreeClassifier(max_depth=8, criterion='entropy')
    decision_tree.fit(X_train, y_train)

    # Evaluate the decision tree
    accuracy = decision_tree.score(X_test, y_test)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")

    # Print feature importances
    feature_importances = decision_tree.feature_importances_
    feature_names = [f"Cell_{i // 5}_{i % 5}" for i in range(25)]
    for name, importance in zip(feature_names, feature_importances):
        print(f"Feature: {name}, Importance: {importance:.4f}")

    return decision_tree


def evaluate_model(decision_tree, X, y):
    scores = cross_val_score(decision_tree, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")


def create_decision_tree(FILENAME):
    # Load data from the file
    windows, risks = load_test_result(FILENAME)

    # Prepare the dataset
    X, y = prepare_dataset(windows, risks)

    # Train the decision tree
    decision_tree = train_decision_tree(X, y)
    evaluate_model(decision_tree, X, y)

    return decision_tree


if __name__ == '__main__':
    os.makedirs('DECISION_TREES', exist_ok=True)
    FILENAME = os.path.join('RESULTS_TEST', f'SMP_TestResult_{BOARD_SIZE}_{ITERATIONS}_{MINES}.txt')
    decision_tree = create_decision_tree(FILENAME)
    
    # Save the model to a file
    joblib.dump(decision_tree, os.path.join('DECISION_TREES', f'DecisionTree_{BOARD_SIZE}_{ITERATIONS}_{MINES}.pkl'), compress=3)  # Using joblib with compression level 3
