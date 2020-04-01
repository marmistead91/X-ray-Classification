import glob
import pandas as pd

def image_counter(file):
    return len(glob.glob1(file, "*.jpeg"))

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