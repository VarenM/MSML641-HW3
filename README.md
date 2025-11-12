# MSML641: Homework 3 - Sentiment Analysis

## *Github Repo: https://github.com/VarenM/MSML641-HW3*

## Setup

Ensure Python is installed correctly (any version past 3.10 should work)

Optionally, create a virtual environment (venv) using:
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

### OR

You can also run this code in a Google Colab notebook (ensure to use T4-Python runtime) to speed up training

first upload the entire project zip as a zip called `project` (containing folders `data` and `src`)

ensure this is the zip architecture:

`project.zip`:

├── data/

│   ├── train/

│   ├── test/

├── src/

│   ├── preprocess.py

│   ├── models.py

│   ├── train.py

│   ├── evaluate.py

│   └── utils.py

After uploading the zip folder to Colab's runtime, use (ensures the two folders are created directly in content and not within a subfolder):
```bash=
!unzip project.zip -d /content
```

## How to Run the Code

If running locally, first, ensure the current working directory is the main project folder (not within src, so remember to call each file with `src/` in front)

Start by running `utils.py`.
```bash=
python src/utils.py
```

### OR

Otherwise if running on Colab use:
```bash=
%run src/utils.py
```

Next run `models.py` (using the command structure shown above).

Then preprocess the data using `preprocess.py` (note, data is already preprocessed, but if you would like to run your own preprocessing, delete the `processed` folder in `data`).

Once you have obtained the preprocessed data (located in `data/processed/`), start training, by running `train.py`.

Finally, to view the best performing model for each method (RNN, LSTM, Bidirectional LSTM) use `evaluate.py`.

If using Colab and would like to download all created files use the following commands:
```bash=
!zip -r processed.zip data/processed
!zip -r results.zip results
```

## Expected Runtimes/Outputs

* All training was completed in a T4 Colab using:

    `Hardware Info: {'platform': 'Linux-6.6.105+-x86_64-with-glibc2.35', 'cpu': 'x86_64', 'ram_gb': 12.671436309814453, 'gpu': 'Tesla T4'}`

`utils.py` = takes under 10 seconds on local machine

Output: prints hardware description (`Hardware Info` seen above)

`models.py` = takes under 10 seconds on local machine

Output: None; it simply sets up the methods to obtain different models

`preprocess.py` = takes about a minute on a local machine (in Colab it runs in seconds)

Output: checks if a tokenizer already exists in `data/processed` and if not creates a tokenizer and tokenizes/pads train/validate/test data. It also prints the location of all saved tokenized files and a report summary of preprocessed data statistics (e.g., avg. review length, vocab size)

`train.py` = takes a little more than an hour on Colab (around 3 secs per epoch) (took up to 2-3 hours my local machine)

Output: prints all combinations of each model with defined parameters and their training epochs (currently set to 10 epochs in `train.py`). Once training is completed, it then builds the plots for measuring Accuracy/F1 vs. Sequence Length and Training Loss vs. Epochs (for best and worst models).

`evaluate.py` = takes under 10 seconds on local machine

Output: Prints the evaluation metrics (accuracy, f1) and specific configs for each best performing model in RNN, LSTM and Bidirectional LSTM