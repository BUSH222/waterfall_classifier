# Waterfall Classifier

A machine learning project to classify satnogs waterfalls (whether they have signal or not)

## DATASET:
~40k images with a signal and ~40k images without a signal

## GOAL:
0.96 accuracy (24/25 classified correctly) OR greater than human accuracy

Since the total satnogs dataset is very big (~13 million observations, out of them at least 10 million with waterfalls), we can calculate the human accuracy, while making use of the normal approximation of the data as per the CLT.\
Assuming:
$$E = 0.01 \text{ (margin of error)}\\
z=1.96\text{ (z-score for the confidence level at } \alpha=0.95 \text{)}\\
p=0.5\text{ (human accuracy, unknown so far)}\\
n - \text{number of samples taken}\\
m - \text{number of misclassified samples out of the total number of samples n}\\
\hat{p} - \text { real human accuracy}$$
We can deduce a minimum sample size to find the true human accuracy p:
$$n=\frac{z^2p(1-p)}{E^2}=\frac{1.96*0.5*0.5}{0.01^2}=9604$$
Consequently, 9604 (I will use ~10000 since experts are imperfect) samples need to be reviewed to estimate the human accuracy. Then the true human accuracy is simply:
$$\hat{p}=1-\frac{m}{n}$$

This number will be estimated in **this branch**

If my model scores above the human margin of error or 0.96 (arbitrary value), I would consider that to be a success.

## Manual labeling web app (FastAPI)

This repo includes a small FastAPI website to show waterfall images one-at-a-time in random order, let an authenticated expert label them (signal / no signal), and compute accuracy vs ground truth.

### 1) Put images somewhere

Default location:

- `data/with_signal/` and `data/without_signal/` (subfolders allowed)

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`.

You can override with:

- `WF_IMAGES_DIR=/absolute/path/to/dataset_root`

Expected layout:

- `WF_IMAGES_DIR/with_signal/*.png`
- `WF_IMAGES_DIR/without_signal/*.png`

### 2) Provide ground-truth labels (recommended)

If your dataset is already split into `with_signal/` and `without_signal/`, ground-truth is inferred automatically and you can skip this step.

Set:

- `WF_LABELS_FILE=/absolute/path/to/labels.csv`

CSV formats supported (header optional):

- `filename,label`
- `filename,has_signal`

Where label can be `1/0`, `signal/no_signal`, `with_signal/without_signal`, `true/false`, etc.

`filename` should match the image path relative to the images directory (including subfolders if any).

If you don’t provide labels, the UI still collects decisions, but “correct/incorrect” will be `unknown`.

### 3) Configure users (rudimentary auth)

Set a comma-separated list of `username:password` pairs:

- `WF_USERS=alice:pass,bob:pass2`

Also set a session signing key:

- `WF_SECRET_KEY=some-long-random-string`

### 4) Run

Install deps:

- `pip install -r requirements.txt`

Start server:

- `python main.py`

Open:

- http://127.0.0.1:8000

### APIs

All APIs require being logged-in (session cookie).

- `GET /api/stats` – totals + accuracy (if labels loaded)
- `GET /api/correct?limit=500` – filenames labeled correctly
- `GET /api/misidentified?limit=500` – filenames labeled incorrectly