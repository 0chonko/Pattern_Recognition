# import libraries and dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import scale
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import cv2
from PIL import Image

# read in data
original = pd.read_csv('data/mnist.csv')
original_digits = original.values[:, 1:]
original_labels = original.values[:, 0]
print(original.shape)

# set image pixel size and plot the first of the digits
plt.imshow(original_digits[0].reshape(28, 28))
plt.show()

# =============================================================================
# Caculating the majority class
# =============================================================================

# checking and plotting class distribution
label_counter = Counter(original_labels)
digits_distribution = original['label'].value_counts()
digits_distribution.plot(kind='bar')
plt.xlabel("Digits labels")
plt.ylabel("Frequency")
plt.show()

# show majority class
print(
    f"The majority class label is {label_counter.most_common(1)[0][0]} with {label_counter.most_common(1)[0][1]} samples.")

# use a DummyClassifier to predict majority class
dummy_model = DummyClassifier(strategy="most_frequent").fit(scale(original_digits), original_labels)
dummy_preds = dummy_model.predict(original_digits)
print(
    f"Accuracy for the majority class dummy classifier: {round(dummy_model.score(original_digits, original_labels), 3)}")


# =============================================================================
# MNIST data preprocessing
# =============================================================================

# function to eliminate pixels with constant intensity values (black & white)
def remove_constant_pixels(mnist_data):
    """Removes from the images the pixels that have a constant intensity value,
    either always black (0) or white (255)
    Returns the cleared dataset"""

    new_mnist_data = mnist_data.loc[:]

    # remove pixels with max value == 0 (black pixels)
    for col in mnist_data:
        if new_mnist_data[col].max() == 0:
            new_mnist_data.drop(columns=[col], inplace=True)

    # remove pixels with min value == 255 (white pixels)
    for col in new_mnist_data:
        if new_mnist_data[col].min() == 255:
            new_mnist_data.drop(columns=[col], inplace=True)

    return new_mnist_data


# get new data
new = remove_constant_pixels(original)
new_digits = new.values[:, 1:]
new_labels = new.values[:, 0]

print(new.shape)

# =============================================================================
# Create ink feature and fit a Logistic Regression model to it
# =============================================================================

# make ink feature vector
ink = np.array([sum(row) for row in new_digits])

# compute mean and stardard deviation for each digit class
ink_mean = [round(np.mean(ink[new_labels == i]), 2) for i in range(10)]
ink_std = [round(np.std(ink[new_labels == i]), 2) for i in range(10)]

# make dict where to store the means and stds for each class
means_stds_per_class_ink = dict()
for i, mean, std in zip(range(10), ink_mean, ink_std):
    means_stds_per_class_ink[f"Digit {i}"] = {"Mean": mean, "SD": std}

plot_ink = pd.DataFrame(means_stds_per_class_ink, index=["Mean", "SD"])
print(plot_ink)

# plot ink values
plt.errorbar(range(10), ink_mean, ink_std, linestyle='none', marker='^', capsize=3)
plt.xlabel("Digits labels")
plt.ylabel("Mean and SD for ink feature")
plt.show()

# scale data to have zero mean and unit variance
ink = scale(ink)

# test train split; the reshape is necessary to call LogisticRegression() with a single feature
ink_x_train, ink_x_test, ink_y_train, ink_y_test = train_test_split(ink.reshape(-1, 1), new_labels, test_size=0.25)

# create and fit LR model
ink_model = LogisticRegression().fit(ink_x_train, ink_y_train)

# models' predictions on test data
ink_preds = ink_model.predict(ink_x_test)

# visualize barplot of metrics's performance on LR on ink feature
ink_metrics = {"accuracy": accuracy_score(ink_y_test, ink_preds),
               "f1-score": f1_score(ink_y_test, ink_preds, average="weighted")}
plt.bar(ink_metrics.keys(), ink_metrics.values())
plt.show()

# generate confusion matrix and classification report
ink_cm = confusion_matrix(ink_y_test, ink_preds)
ink_cr = classification_report(ink_y_test, ink_preds)
print(ink_cr)

# =============================================================================
# Caculating the neighbours
# Two functions to calculate the digits
# 1. calc the max horizontal neighbour: input = the list with horizontal neighbours
# 3. function to calc the max vertical neighbour: input = the list with vertical neighbours
# =============================================================================

original_img_size = 28
new_img_size = 26  # 26.627053911388696


def calc_horizontal_neighbour(x, img_size):
    max_neighbourcount = 0
    for row in range(img_size):
        state = 0
        for i in range(img_size):
            if x[img_size * (row - 1) + i] != 0:
                state += 1
            elif state > max_neighbourcount:
                max_neighbourcount = state
                state = 0
    return max_neighbourcount


def calc_vertical_neighbour(x, img_size):
    max_neighbourcount = 0

    # Loop through all columns
    for column in range(img_size):
        state = 0

        # for each row in the digit
        for row in range(img_size):

            # check if digit has ink
            if x[img_size * row + column] != 0:
                state += 1
            elif state > max_neighbourcount:
                max_neighbourcount = state
                state = 0
    return max_neighbourcount


neighbors_v = np.zeros(len(new_digits))
neighbors_h = np.zeros(len(new_digits))


# =============================================================================
# Find intersecction lines (if the holes feature is good enough this becomes obsolete
# =============================================================================
def find_extremes(digit):
    array = np.array(digit, dtype=np.uint8)
    array = np.reshape(array, (28, 28))

    img = Image.fromarray(array, 'L')
    # find contours
    cnts = cv2.findContours(array, cv2.CV_8SC1, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return line_intersection((left, right), (top, bottom))


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return 14, 14

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


intersection_v = np.zeros(len(original_digits))
intersection_h = np.zeros(len(original_digits))

for i in range(len(original_digits)):
    center = find_extremes(original_digits[i])
    intersection_h[i] = center[0]
    intersection_v[i] = center[1]

scale(intersection_h)
scale(intersection_v)
intersection = np.column_stack((intersection_h, intersection_v))

# =============================================================================
# Find holes
# =============================================================================
holes = np.zeros(len(original_digits))
for i in range(len(original_digits)):
    # set pixel array to 2D
    array = np.array(original_digits[i], dtype=np.uint8)
    array = np.reshape(array, (28, 28))
    # retrieve hierarchy and contours
    contours, hierarchy = cv2.findContours(array, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes[i] = len([contours[j] for j in range(len(contours)) if hierarchy[0][j][3] >= 0])
# compute mean
holes_mean = [np.mean(holes[original_labels == i]) for i in range(10)]
# compute standard deviation
holes_std = [np.std(holes[original_labels == i]) for i in range(10)]
print(holes_mean)
print(holes_std)
scale(holes)

# create neighbors feature by multiplying the vertical and horizontal max lengths and scaling to -1,1
for i in range(len(new_digits)):
    neighbors_h[i] = calc_horizontal_neighbour(new_digits[i], new_img_size)
    neighbors_v[i] = calc_vertical_neighbour(new_digits[i], new_img_size)

neighbors = np.column_stack((neighbors_v, neighbors_h))

# compute mean and stardard deviation for each digit class
NB_mean = [round(np.mean(neighbors[new_labels == i]), 2) for i in range(10)]
NB_std = [round(np.std(neighbors[new_labels == i]), 2) for i in range(10)]

# make dict where to store the means and stds for each class
means_stds_per_class_NB = dict()
for i, mean, std in zip(range(10), NB_mean, NB_std):
    means_stds_per_class_NB[f"Digit {i}"] = {"Mean": mean, "SD": std}

plot_NB = pd.DataFrame(means_stds_per_class_NB, index=["Mean", "SD"])
print(plot_NB)

# plot neighbors values
plt.errorbar(range(10), NB_mean, NB_std, linestyle='none', marker='^', capsize=3)
plt.xlabel("Digits labels")
plt.ylabel("Mean and SD for neighbours feature")
plt.show()

# =============================================================================
# Model training and evaluation
# =============================================================================

# scale data to have zero mean and unit variance
scale(neighbors)

# test train split with neighbors features only
# the reshape is necessary to call LogisticRegression() with a single feature
# neighbors_x_train, neighbors_x_test, neighbors_y_train, neighbors_y_test = train_test_split(neighbors.reshape(-1, 1), new_labels, test_size=0.25)

scale(intersection)

neighbors_x_train, neighbors_x_test, neighbors_y_train, neighbors_y_test = train_test_split(neighbors, original_labels,
                                                                                            test_size=0.25)
# test train split with both features

combined_features = np.column_stack((ink, neighbors, intersection))
combined_features = np.column_stack((combined_features, holes))

both_x_train, both_x_test, both_y_train, both_y_test = train_test_split(combined_features, original_labels,
                                                                        test_size=0.25)

# create models
neighbors_model = LogisticRegression().fit(neighbors_x_train, neighbors_y_train)
both_model = LogisticRegression().fit(both_x_train, both_y_train)

# models' predictions on test data
neighbors_preds = neighbors_model.predict(neighbors_x_test)
both_preds = both_model.predict(both_x_test)

# =============================================================================
# Model performance evaluation and visualization
# =============================================================================

# generate confusion matrices
neighbors_cm = confusion_matrix(neighbors_y_test, neighbors_preds)
both_cm = confusion_matrix(both_y_test, both_preds)

# generate classification reports
neighbors_cr = classification_report(neighbors_y_test, neighbors_preds)
both_cr = classification_report(both_y_test, both_preds)

# save metrics in a dict for later plotting
metrics = {"accuracy": {"ink": accuracy_score(ink_y_test, ink_preds),
                        "neighbors": accuracy_score(neighbors_y_test, neighbors_preds),
                        "both": accuracy_score(both_y_test, both_preds)},
           "f1-score": {"ink": f1_score(ink_y_test, ink_preds, average="weighted"),
                        "neighbors": f1_score(neighbors_y_test, neighbors_preds, average="weighted"),
                        "both": f1_score(both_y_test, both_preds, average="weighted")}}

# plot metrics to compare performance
mdf = pd.DataFrame(metrics)
mdf.plot(kind="bar")
plt.show()

# visualize ink confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(ink_cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='rocket')
plt.ylabel('Actual label with ink feature')
plt.xlabel('Predicted label with ink feature')
plt.show()

# visualize neighbors confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(neighbors_cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='rocket')
plt.ylabel('Actual label with neighbors feature')
plt.xlabel('Predicted label with neighbours feature')
plt.show()

# visualize both features confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(both_cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='rocket')
plt.ylabel('Actual label with both features')
plt.xlabel('Predicted label with both features')
plt.show()

# =============================================================================
# Fit models only using pixel values
# =============================================================================

# first resize images to 14x14 pixels instead of 28x28
resized_digits = [cv2.resize(img.astype("uint8"), dsize=(14, 14)) for img in new_digits]
resized_digits = np.array([img.reshape((196,)) for img in resized_digits])
print(new_digits[0].shape)
print(resized_digits[0].shape)

# set image pixel size and plot the first of the resized digits
img_size = 14
plt.imshow(resized_digits[0].reshape(img_size, img_size))
plt.show()

# create new splits with 5000 samples to use for training and remaining 37000 for testing
pixels_train, pixels_test, labels_train, labels_test = train_test_split(scale(resized_digits), new_labels,
                                                                        train_size=5000, test_size=37000)

# fit models
pixels_LR = LogisticRegression().fit(pixels_train, labels_train)
pixels_MLP = MLPClassifier().fit(pixels_train, labels_train)
pixels_SVC = SVC().fit(pixels_train, labels_train)

# get models' predictions
pixels_LR_preds = pixels_LR.predict(pixels_test)
pixels_MLP_preds = pixels_MLP.predict(pixels_test)
pixels_SVC_preds = pixels_SVC.predict(pixels_test)
print(f"""Models trained on 5000 samples and tested on remaining 37000 samples
      LR = {accuracy_score(labels_test, pixels_LR_preds)}
      MLP = {accuracy_score(labels_test, pixels_MLP_preds)}
      SVC = {accuracy_score(labels_test, pixels_SVC_preds)}""")

# plot metrics to compare performance
accuracies = {"LR": accuracy_score(labels_test, pixels_LR_preds),
              "MLP": accuracy_score(labels_test, pixels_MLP_preds),
              "SVC": accuracy_score(labels_test, pixels_SVC_preds)}
plt.bar(accuracies.keys(), accuracies.values())
plt.show()

precisions = {"LR": precision_score(labels_test, pixels_LR_preds, average='micro'),
              "MLP": precision_score(labels_test, pixels_MLP_preds, average='micro'),
              "SVC": precision_score(labels_test, pixels_SVC_preds, average='micro')}

recall = {"LR": recall_score(labels_test, pixels_LR_preds, average='micro'),
          "MLP": recall_score(labels_test, pixels_MLP_preds, average='micro'),
          "SVC": recall_score(labels_test, pixels_SVC_preds, average='micro')}

f1score = {"LR": f1_score(labels_test, pixels_LR_preds, average='micro'),
           "MLP": f1_score(labels_test, pixels_MLP_preds, average='micro'),
           "SVC": f1_score(labels_test, pixels_SVC_preds, average='micro')}

exit()

# =============================================================================
# Plotting the confusion matrices
#
# =============================================================================
LR_CR = confusion_matrix(labels_test, pixels_LR_preds)
MLP_CR = confusion_matrix(labels_test, pixels_MLP_preds)
SVC_CR = confusion_matrix(labels_test, pixels_SVC_preds)

# visualize both features confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(LR_CR, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='rocket')
plt.ylabel('Actual label')
plt.xlabel('Predicted label by MRL')
plt.show()

# visualize both features confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(MLP_CR, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='rocket')
plt.ylabel('Actual label')
plt.xlabel('Predicted label by MLP')
plt.show()

# visualize both features confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(SVC_CR, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='rocket')
plt.ylabel('Actual label')
plt.xlabel('Predicted label by SVC')
plt.show()

# =============================================================================
# Statistical test
# =============================================================================

from mlxtend.evaluate import paired_ttest_5x2cv

t, p = paired_ttest_5x2cv(estimator1=pixels_SVC, estimator2=pixels_MLP, X=pixels_test, y=labels_test)

if p <= 0.05:
    print('Difference between mean performance is probably real')
else:
    print('Algorithms probably have the same performance')

print(str(p) + 'p of SVC vs MLP')
print(str(t) + 't of SVC vs MLP')

t2, p2 = paired_ttest_5x2cv(estimator1=pixels_LR, estimator2=pixels_MLP, X=pixels_test, y=labels_test)

if p2 <= 0.05:
    print('Difference between mean performance is probably real')
else:
    print('Algorithms probably have the same performance')

print(str(p2) + 'p of LR vs MLP')
print(str(t2) + 't of LR vs MLP')

t3, p3 = paired_ttest_5x2cv(estimator1=pixels_SVC, estimator2=pixels_LR, X=pixels_test, y=labels_test)

if p3 <= 0.05:
    print('Difference between mean performance is probably real')
else:
    print('Algorithms probably have the same performance')

print(str(p3) + 'p of SVC vs LR')
print(str(t3) + 't of SVC vs LR')

exit('completed ttests')

###############################################################################
# Cross Validation hyperparameters and model selection
# WARNING: the code below will take roughly 50 minutes to create, train, evaluate and compare 100 different models
###############################################################################

import datetime
from numpy import mean, std
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

begin_time = datetime.datetime.now()

# configure the cross-validation procedure
cv_inner = KFold(n_splits=5, shuffle=True)

# define the models
LR = LogisticRegression()
MLP = MLPClassifier()
SVC = SVC()

# define search space for LR, trying 5 * 2 * 2 ==> 20 total evaluations
LR_space = {"C": [10, 0.1]}

# define search space for MLP, trying 5 * (2 * 2) * 2 ==> 40 total evaluations
MLP_space = {"alpha": [0.001, 0.00001], "tol": [1e-3, 1e-5]}

# define search space for SVC, trying 5 * (2 * 2) * 2 ==> 40 total evaluations
SVC_space = {"C": [10, 0.1], "tol": [1e-3, 1e-5]}

# define search for the models
LR_search = GridSearchCV(LR, LR_space, scoring='accuracy', n_jobs=-1, cv=cv_inner, refit=True)
MLP_search = GridSearchCV(MLP, MLP_space, scoring='accuracy', n_jobs=-1, cv=cv_inner, refit=True)
SVC_search = GridSearchCV(SVC, SVC_space, scoring='accuracy', n_jobs=-1, cv=cv_inner, refit=True)

# configure the cross-validation procedure
cv_outer = KFold(n_splits=2, shuffle=True)

# execute the nested cross-validation
LR_scores = cross_val_score(LR_search, scale(new_digits), new_labels, scoring='accuracy', cv=cv_outer, n_jobs=-1)
MLP_scores = cross_val_score(MLP_search, scale(new_digits), new_labels, scoring='accuracy', cv=cv_outer, n_jobs=-1)
SVC_scores = cross_val_score(SVC_search, scale(new_digits), new_labels, scoring='accuracy', cv=cv_outer, n_jobs=-1)

# report performances
print('LR Accuracy: %.3f (%.3f)' % (mean(LR_scores), std(LR_scores)))
print('MLP Accuracy: %.3f (%.3f)' % (mean(MLP_scores), std(MLP_scores)))
print('SVC Accuracy: %.3f (%.3f)' % (mean(SVC_scores), std(SVC_scores)))

print(datetime.datetime.now() - begin_time)

