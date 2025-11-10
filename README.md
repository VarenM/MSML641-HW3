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

# OR

You can also run this code in a Google Colab notebook (ensure to use T4-Python runtime) to speed up training

first upload the entire project zip (containing folders `data` and `src`)

ensure this is the zip architecture:

project.zip:

├── data/

│   ├── train/

│   ├── test/

├── src/

│   ├── preprocess.py

│   ├── models.py

│   ├── train.py

│   ├── evaluate.py

│   └── utils.py

After uploading the zip folder to Colab's runtime, use:
```bash=
!unzip project.zip -d /content
```

## How to Run the Code

If running locally, first, ensure the current working directory is the main project folder (not within src, so remember to call each file with `src/` in front)

Start by running `utils.py`.
```bash=
python src/utils.py
```

# OR

Otherwise if running on Colab use:
```bash=
%run src/utils.py
```

Next run `models.py`.

Then preprocess the data using `preprocess.py`.

Once you have obtained the preprocessed data (located in `data/processed/`), start training, by running `train.py`.

Finally, to view the best model's performance use `evaluate.py`.

## Expected Runtimes/Outputs

`preprocess.py` = takes about a minute on a local machine (in colab it runs in seconds)

`models.py` = takes about 5 seconds on local machine

`utils.py` = takes about 3 seconds on local machine

`train.py` = takes a little more than an hour on Colab machine (around 3 secs per epoch) (around 2 hours on local)

`evaluate.py` = takes about 5 seconds on local machine