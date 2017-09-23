import matplotlib.image as mpimg
import sklearn
import config as cf
from preprocess import *
from augmentation import *

def data_generator(df, batch_size, data_augmentation):
    images = np.zeros([batch_size, cf.IMAGE_HEIGHT, cf.IMAGE_WIDTH, cf.IMAGE_CHANNELS])
    steering_angles = np.zeros(batch_size)
    X = df[['center', 'left', 'right']].values
    y = df['steering'].values
    steering_angle_cal = [0, 0.2, -0.2]
    index = 0
    df_len = len(df)

    while True:
        i = 0

        while i < batch_size:
            index %= df_len
            if index == 0:
                df = sklearn.utils.shuffle(df)
                X = df[['center', 'left', 'right']].values
                y = df['steering'].values

            choice = np.random.choice(3)
            image = mpimg.imread(X[index][choice])
            steering_angle = y[index] + steering_angle_cal[choice]
            if data_augmentation and np.random.rand() < 0.5:
                image, steering_angle = augument(image, steering_angle)

            images[i] = preprocess(image)
            steering_angles[i] = steering_angle
            i += 1
            index += 1

        yield sklearn.utils.shuffle(np.array(images), np.array(steering_angles))
