# MSML641: Homework 3 - Sentiment Analysis

## Setup

Ensure Python is installed correctly

Create a virtual environment (venv) using:
```bash=
python -m venv venv

source venv/bin/activate       # Mac/Linux
# OR
venv\Scripts\activate          # Windows
```

Then install all Python dependencies from requirements.txt using (can be done without a venv):
```bash=
pip install -r requirements.txt
```

## How to Run the Code

First, ensure the current working directory is the main project folder (not within src, so remember to call each file with `src/` in front)

Start by preprocessing the data using `preprocess.py`.
```bash=
python src/preprocess.py
```

Once you have obtained the preprocessed data (located in `data/processed/`), run `models.py`

Next run `utils.py`.

Start training, by running `train.py` after the above steps have been completed.

Finally, to view the best model's performance use `evaluate.py`.

## Expected Runtimes/Outputs

`preprocess.py` = takes about a minute on a local machine (in colab it runs in seconds)

`models.py` = takes about 5 seconds on local machine

`utils.py` = takes about 3 seconds on local machine

`train.py` = takes about 1 hour on local machine (around 12 secs per epoch)

`evaluate.py` = takes about 5 seconds on local machine