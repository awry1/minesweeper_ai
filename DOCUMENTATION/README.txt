README.txt - What's in the folders?

1) The *analytical* folder contains datasets (on which the models were/are trained) and logs of win ratios. The datasets should ideally have 1000 iterations (cries in train CNN).

2) The *neural networks* folder stores the "trained" models (300 epochs, the same learning rates, gamma, etc.), training and testing logs, as well as decision trees (of course, only 5x5). Everything is organized into the appropriate folders.
IMPORTANT: The number of iterations in the folder name refers to the size of the dataset used for training, not the number of iterations in the `solve.py` program.

NOTES:
1) Remember that the board size must be the same for both training and testing.

2) Where possible, use the seed 'alamakota' to get results that are as similar as possible.
Unless it's for testing, in which case it’s better to test on a different dataset than the one used for training. I suggest using the seed 'kotmaale' for this.