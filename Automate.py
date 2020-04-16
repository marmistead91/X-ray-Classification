import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show
import numpy as np
import itertools
import PIL
from PIL import Image
import math
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras import models, layers

def image_counter(file):
    """

    :param file: file to be counted
    :return: amount of images in each file
    """
    return len(glob.glob1(file, "*.jpeg"))

def height_width(file):
    """

    :param file: file of images
    :return: dataframe of height and width of images
    """
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
    """

    :param filenorm: files of norm xrays
    :param filesick: files of pneumonia xrays
    :return: dataframe with images concated
    """
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
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
#                  verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

def visualize_training_results(results):
    """

    :param results: results from model that was run
    :return: plot of val_loss and val_acc
    """
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

def extracting_features_map(img_path, model):
    """

    :param img_path: image to use for feature extraction
    :param model: model used to extract features
    :return: feature extraction of each layer from model
    """
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract model layer outputs
    layer_outputs = [layer.output for layer in model.layers[:8]]

    # Create a model for displaying the feature maps
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(x)

    # Extract Layer Names for Labelling
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    total_features = sum([a.shape[-1] for a in activations])
    total_features

    n_cols = 16
    n_rows = math.ceil(total_features / n_cols)

    iteration = 0
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols, n_rows * 1.5))

    for layer_n, layer_activation in enumerate(activations):
        n_channels = layer_activation.shape[-1]
        for ch_idx in range(n_channels):
            row = iteration // n_cols
            column = iteration % n_cols

            ax = axes[row, column]

            channel_image = layer_activation[0,
                            :,
                            :,
                            ch_idx]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            ax.imshow(channel_image, aspect='auto', cmap='viridis')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if ch_idx == 0:
                ax.set_title(layer_names[layer_n], fontsize=10)
            iteration += 1

    fig.subplots_adjust(hspace=1.25)
    plt.show()

def prec_recall_f1(cm):
    """
    takes in confusion matrix and gives recall, precision and f1 score
    """
    tp = cm[0][0]
    fp = cm[1][0]
    fn = cm[0][1]
    recall = round((tp / (tp + fn)), 2)
    precision = round((tp / (tp + fp)), 2)
    f1 = round((2 * ((precision * recall) / (precision + recall))), 2)
    print(f'Recall : {recall}')
    print(f'Precision : {precision}')
    print(f'F1 : {f1}')


def plot(data_generator):
    """
    Plots 4 images generated by an object of the ImageDataGenerator class.
    """
    image = imread("chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg")

    # Creating a dataset which contains just one image.
    images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    data_generator.fit(images)
    image_iterator = data_generator.flow(images)

    # Plot the images given by the iterator
    fig, rows = subplots(nrows=1, ncols=4, figsize=(18, 18))
    for row in rows:
        row.imshow(image_iterator.next()[0].astype('int'))
        row.axis('off')
    show()