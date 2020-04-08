import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import PIL
from PIL import Image

def image_counter(file):
    return len(glob.glob1(file, "*.jpeg"))

def height_width(file):
    x = glob.glob1(file, "*.jpeg")
    width = []
    height = []
    for img in x:
        image = PIL.Image.open(file+img)
        w, h = image.size
        width.append(w)
        height.append(h)
    wseries = pd.Series(width)
    hseries = pd.Series(height)
    return pd.concat([wseries, hseries], axis=1)

def images_to_df(filenorm, filesick):
    # Get the list of all the images
    normal_cases = glob.glob1(filenorm, '*.jpeg')
    pneumonia_cases = glob.glob1(filesick, '*.jpeg')

    # An empty list. We will insert the data into this list in (img_path, label) format
    train_data = []

    # Go through all the normal cases. The label for these cases will be 0
    for img in normal_cases:
        train_data.append((img, 0))

    # Go through all the pneumonia cases. The label for these cases will be 1
    for img in pneumonia_cases:
        train_data.append((img, 1))

    # Get a pandas dataframe from the data we have in our list
    train_data = pd.DataFrame(train_data, columns=['image', 'label'], index=None)

    # Shuffle the data
    train_data = train_data.sample(frac=1.).reset_index(drop=True)
    return train_data

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest',aspect='equal', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
#                  verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def visualize_training_results(results):
    history = results.history
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()