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

This number will be estimated in another branch

If my model scores above the human margin of error or 0.96 (arbitrary value), I would consider that to be a success.


## Attempts:
1. Transfer learning
    - data heavily cropped and downscaled in half
    - resnet18 transfer learning
    - 0.9 accuracy
    - Model file not published, ask if needed
2. Transfer learning attempt 2
    - Added a scheduler
    - reduced learning rate slightly
    - changed model to resnet34
    - 0.91 accuracy
    - Model file not published, ask if needed
3. in progress