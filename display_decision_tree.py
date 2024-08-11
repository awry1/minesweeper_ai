import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Constants for quick change
BOARD_SIZE = 10, 10
ITERATIONS = 1000
MINES = 10


def plot_feature_importances(feature_importances, feature_names):
    FILENAME = os.path.join('DECISION_TREES',f'FeatureImportance_{BOARD_SIZE}_{ITERATIONS}_{MINES}.png')

    indices = np.argsort(feature_importances)
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(feature_importances)), feature_importances[indices], align='center')
    plt.yticks(range(len(feature_importances)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.savefig(FILENAME, bbox_inches='tight')
    plt.show()


def visualize_decision_tree(decision_tree):
    feature_names = [f"Cell_{i // 5}_{i % 5}" for i in range(25)]
    plot_feature_importances(decision_tree.feature_importances_, feature_names)

    FILENAME = os.path.join('DECISION_TREES', f'DecisionTree_{BOARD_SIZE}_{ITERATIONS}_{MINES}.png')

    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree,
              filled=True,
              feature_names=feature_names,
              class_names=["not a mine", "probably not a mine", "probably mine", "mine"],
              max_depth=2,
              rounded=True)
    plt.title('Decision Tree Visualization')
    plt.savefig(FILENAME, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Load the model
    FILENAME = os.path.join('DECISION_TREES', f'DecisionTree_{BOARD_SIZE}_{ITERATIONS}_{MINES}.pkl')
    decision_tree = joblib.load(FILENAME)

    visualize_decision_tree(decision_tree)
