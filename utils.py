import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os
import timeit
import pathlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def plot_histogram(data, label=[]):
    # plt.figure(figsize=(16,7))
    plt.hist(data, bins=100)
    plt.xlabel('Steering angle')
    plt.ylabel('Number of samples')
    if label != []:
        # plt.legend([label], fontsize='xx-large')
        plt.legend([label])
    plt.show()


def load_data(data_dir):
    images = []
    steering_angles = []
    udacity_data_dir = None
    # HOME = os.getenv('HOME')
    # path = pathlib.Path(HOME + '/dataset')
    path = pathlib.Path(data_dir)
    driving_log = pd.DataFrame([])
    df_data = pd.DataFrame([])

    path = pathlib.Path(data_dir)
    for folder in path.iterdir():
        start_time = timeit.default_timer()
        if folder.is_file():
            fname = os.path.split(folder.as_posix())[1]
            file = folder.as_posix()
            folder = folder.parent
            if fname != 'driving_log.csv':
                continue
        elif folder.is_dir():
            file = folder.joinpath('driving_log.csv').as_posix()

        if not os.path.isfile(file):
            continue

        df = pd.read_csv(file)
        headers = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        if headers == df.columns.tolist():
            udacity_data_dir = folder.as_posix()
            df = pd.read_csv(file)
            df_obj = df.select_dtypes(['object'])
            df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

            df['center'] = udacity_data_dir + '/' +df['center']
            df['left'] = udacity_data_dir + '/' +df['left']
            df['right'] = udacity_data_dir + '/' +df['right']

            image_paths = df[['center', 'left', 'right']].values
            angles = df['steering'].values
        else:
            df = pd.read_csv(file, names=headers)
            df_obj = df.select_dtypes(['object'])
            df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

        driving_log = pd.concat([driving_log, df])

    df = driving_log
    image_paths = df[['center', 'left', 'right']].values
    angles = df['steering'].values
    X = df

    df = sklearn.utils.shuffle(df)
    df_train, df_valid = train_test_split(X, test_size=0.2, random_state=0)

    return df_train, df_valid

def plot_history(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
