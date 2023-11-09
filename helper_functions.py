### We create a bunch of helpful functions throughout the course.
### Storing them here so they're easily accessible.

import tensorflow as tf

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
  
# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
  
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


# Plot loss curves of one or more history object, handles fine tune plots as well
import matplotlib.pyplot as plt

def plot_loss_curves(history, metrics=['accuracy'], finetune=False):
    """
    Plot loss and specified metrics for a TensorFlow model History object or a list of History objects.

    Args:
      history: TensorFlow model History object or a list of History objects. When finetune is True, provide at least two history objects.
      metrics: List of metrics to plot (default is ['accuracy']).
      finetune: Boolean indicating whether the histories are connected (default is False).

    Returns:
      None
    """
    # Check for valid input when finetune is True
    if finetune and len(history) < 2:
        raise ValueError("When 'finetune' is True, provide at least two history objects.")

    # Convert a single history object to a list
    if not isinstance(history, (list, tuple)):
        history = [history]

    # Determine the total number of epochs for all histories
    total_epochs = sum([len(hist.history['loss']) for hist in history])

    if finetune:
        # Combine histories
        combined_history = {
            'loss': [],
            'val_loss': [],
        }

        for metric in metrics:
            combined_history[metric] = []
            combined_history['val_' + metric] = []

        for idx, hist in enumerate(history):
            initial_epochs = sum([len(h.history['loss']) for h in history[:idx]])
            combined_history['loss'] += hist.history['loss']
            combined_history['val_loss'] += hist.history['val_loss']

            for metric in metrics:
                combined_history[metric] += hist.history[metric]
                combined_history['val_' + metric] += hist.history['val_' + metric]

        # Create a single subplot for combined history
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Initialize the epoch counter
        epochs = range(1, total_epochs + 1)

        # Plot combined loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, combined_history['loss'], label='Training Loss')
        plt.plot(epochs, combined_history['val_loss'], label='Validation Loss')
        plt.axvline(x=initial_epochs, color='gray', linestyle='--')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        # Plot specified metrics for combined history
        plt.subplot(1, 2, 2)
        for metric in metrics:
            if metric in combined_history:
                plt.plot(epochs, combined_history[metric], label=f'Training {metric}')
                plt.plot(epochs, combined_history['val_' + metric], label=f'Validation {metric}')
        plt.axvline(x=initial_epochs, color='gray', linestyle='--')
        plt.title('Metrics')
        plt.xlabel('Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        # Original logic for plotting individual histories
        num_histories = len(history)
        if num_histories == 1:
            # Add a single subplot for one history
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        else:
            fig, axs = plt.subplots(num_histories, 2, figsize=(12, 6*num_histories))

        # Initialize the epoch counter
        epoch_counter = 0

        for idx, hist in enumerate(history):
            # Update the epoch counter
            epochs = range(epoch_counter, epoch_counter + len(hist.history['loss']))
            epoch_counter += len(hist.history['loss'])

            # When finetune is False, use the appropriate subplot for each history
            if num_histories == 1:
                axs[0].plot(epochs, hist.history['loss'], label=f'Training Loss (Run {idx+1})')
                axs[0].plot(epochs, hist.history['val_loss'], label=f'Validation Loss (Run {idx+1})')
                axs[0].set_title(f'Loss')
                axs[0].set_xlabel('Epochs')
                axs[0].legend()

                # Plot specified metrics for each history
                for metric in metrics:
                    if metric in hist.history:
                        axs[1].plot(epochs, hist.history[metric], label=f'Training {metric} (Run {idx+1})')
                        axs[1].plot(epochs, hist.history["val_"+metric], label=f'Validation {metric} (Run {idx+1})')
                        axs[1].set_title(metric)
                        axs[1].set_xlabel('Epochs')
                        axs[1].legend()
            else:
                axs[idx, 0].plot(epochs, hist.history['loss'], label=f'Training Loss (Run {idx+1})')
                axs[idx, 0].plot(epochs, hist.history['val_loss'], label=f'Validation Loss (Run {idx+1})')
                axs[idx, 0].set_title(f'Loss')
                axs[idx, 0].set_xlabel('Epochs')
                axs[idx, 0].legend()

                # Plot specified metrics for each history
                for metric in metrics:
                    if metric in hist.history:
                        axs[idx, 1].plot(epochs, hist.history[metric], label=f'Training {metric} (Run {idx+1})')
                        axs[idx, 1].plot(epochs, hist.history["val_"+metric], label=f'Validation {metric} (Run {idx+1})')
                        axs[idx, 1].set_title(metric)
                        axs[idx, 1].set_xlabel('Epochs')
                        axs[idx, 1].legend()

        plt.tight_layout()
        plt.show()
  
# Create function to unzip a zipfile into current working directory 
# (since we're going to be downloading and unzipping a few files)
import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results