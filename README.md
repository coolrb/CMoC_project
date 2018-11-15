# CMoC_project

Final project for CS 328: computational models of cognition. By James Yang and Gabriel Brookman.

We tried to make an LSTM that can generate titles for emails based on email text.

To get the Enron email dataset: download the latest version from https://www.cs.cmu.edu/~./enron/. Then, unzip the file and go to the directory that contains all the email documents. Call `find dir -type f -print0 | xargs -0 -I%%% mv %%% newdir/` to flatten the directory and move all the files to a new directory.

To run our model: process the directory of email .txt files from above by running `data_cleaning.py` with python3, and then use `modeltrainer.py` (also with python3) to train the model.

You should have a subdirectory "saved" in the project directory, which will store the models after they're trained.

The subdirectory "old_version" contains our originally model, as we talk about in the paper. You can run it the same way you can run our main code.
