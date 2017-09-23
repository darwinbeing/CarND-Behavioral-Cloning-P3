from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model
from keras.utils import plot_model
import argparse
import os
import math

from utils import *
from model import *
from generator import *

def train_model(model, df_train, df_valid, epochs=200, batch_size=32):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=1e-4, mode='min')
    tensorBoard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True)

    nb_train_samples = len(df_train)
    nb_validation_samples = len(df_valid)

    steps_per_epoch = math.ceil(1. * nb_train_samples / batch_size)
    validation_steps = math.ceil(1. * nb_validation_samples / batch_size)
    print('steps_per_epoch={}, validation_steps={} epochs={}'.format(steps_per_epoch, validation_steps, epochs))
    if steps_per_epoch <= 0:
        raise AssertionError("Found 0 train samples")
    if validation_steps <= 0:
        raise AssertionError("Found 0 validation samples")

    train_generator = data_generator(df_train, batch_size, True)
    validation_generator = data_generator(df_valid, batch_size, False)

    history_object = model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        # callbacks=[checkpoint, earlyStopping, tensorBoard],
                        callbacks=[checkpoint, tensorBoard],
                        verbose=1)

    return history_object

def main():
    parser = argparse.ArgumentParser(description='CarND-Behavioral-Cloning-P3')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-n', help='number of epochs',      dest='epochs',            type=int,   default=200)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=32)
    args = parser.parse_args()

    np.random.seed(1234)

    df_train, df_valid = load_data(args.data_dir)
    # if os.path.exists('model.h5'):
    #     model = load_model('model.h5')
    #     print('Previous model loaded!')
    # else:
    #     model = get_nvidia_model()

    model = get_nvidia_model()
    plot_model(model, to_file='model.png')

    history = train_model(model, df_train, df_valid, args.epochs, args.batch_size)

    plot_history(history)

if __name__ == '__main__':
    main()
