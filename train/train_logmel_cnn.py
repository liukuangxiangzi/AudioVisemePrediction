import os
from datetime import datetime
import argparse
from keras.layers import Input, LeakyReLU
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.optimizers import adam_v2
from tensorflow.keras.utils import plot_model
from keras.layers import Dense, Flatten, Conv1D
import logging
import h5py

def configure_logging():
    logging.basicConfig(level=logging.INFO, filename='../myapp.log', filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
def load_data(x_train, x_test, y_train, y_test):
    with h5py.File(x_train, 'r') as f1:
        trainx = f1['data'][:]
    with h5py.File(x_test, 'r') as f2:
        testx = f2['data'][:]
    with h5py.File(y_train, 'r') as f3:
        trainy = f3['data'][:]
    with h5py.File(y_test, 'r') as f4:
        testy = f4['data'][:]
    trainx = trainx.reshape(trainx.shape[0] * trainx.shape[1], -1)
    trainx = trainx.reshape(-1, 6, 128)
    testx = testx.reshape(testx.shape[0] * testx.shape[1], -1)
    testx = testx.reshape(-1, 6, 128)
    trainy = trainy.reshape(trainy.shape[0] * trainy.shape[1], -1)
    testy = testy.reshape(testy.shape[0] * testy.shape[1], -1)
    print('trainx', trainx.shape)
    print('testx', testx.shape)
    print('trainy', trainy.shape)
    print('testy', testy.shape)
    return trainx,testx,trainy,testy
def define_model(trainx, resume_training, load_model_dir):
    if not resume_training:
        n_steps, n_features = trainx.shape[1], trainx.shape[2]
        lr =  1e-4
        initializer = 'glorot_uniform'

        # define trained_model
        net_in = Input(shape=(n_steps, n_features))
        l1 = Conv1D(170, 3, 2,
                    padding='same',
                    kernel_initializer=initializer,
                    name='phoneme_Conv1D_1')(net_in)
        lrelu1 = LeakyReLU(0.02)(l1)

        l2 = Conv1D(128, 3, 2,
                    padding='same',
                    kernel_initializer=initializer,
                    name='phoneme_Conv1D_2')(lrelu1)

        lrelu2 = LeakyReLU(0.02)(l2)

        l3 = Conv1D(64, 3, 2,
                    padding='same',
                    kernel_initializer=initializer,
                    name='phoneme_Conv1D_3')(lrelu2)

        lrelu2_2 = LeakyReLU(0.02)(l3)

        flatten = Flatten()(lrelu2_2)

        d1 = Dense(128,
                   kernel_initializer=initializer, name='d1')(flatten)
        lrelu3 = LeakyReLU(0.02)(d1)

        d2 = Dense(64,
                   kernel_initializer=initializer, name='d2')(lrelu3)
        lrelu4 = LeakyReLU(0.02)(d2)

        d3 = Dense(13,
                   kernel_initializer=initializer, name='d_out',activation='softmax')(lrelu4)#

        model = Model(inputs=net_in, outputs=d3)
        model.summary()
        opt = adam_v2.Adam(learning_rate=lr)
        # opt = SGD(learning_rate=lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    else:
        model = load_model(load_model_dir)
        logging.info('Trained_model loaded')
    return model
def train_model(model, trainx, trainy, testx, testy, epochs, batch_size, log_dir, save_model_dir):
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    logging.info('Starting logmel CNN model training')
    history = model.fit(x=trainx,
                        y=trainy,
                        validation_data=(testx, testy),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[tbCallBack],
                        shuffle=True)
    # Plot model structure
    plot_model(model, to_file='../model_logmel_cnn_plot.png', show_shapes=True, show_layer_names=True)
    # Log loss to console and file
    logging.info('Training loss: out={:.4f}'.format(history.history['loss'][-1]))
    logging.info('Validation loss: out={:.4f}'.format(history.history['val_loss'][-1]))
    # Save model to file
    model.save(os.path.join(save_model_dir, 'logmel_cnn_epoch{}_bs{}.h5'.format(epochs, batch_size)))
    logging.info('Model saved to {}'.format(save_model_dir))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-x-data-dir', type=str, default='data/logmel_train_data.h5', help='path to training input data')
    parser.add_argument('--test-x-data-dir', type=str, default='data/logmel_test_data.h5', help='path to test input data')
    parser.add_argument('--train-y-data-dir', type=str, default='data/75_viseme_train_data.h5', help='path to viseme ID training data')
    parser.add_argument('--test-y-data-dir', type=str, default='data/75_viseme_test_data.h5', help='path to viseme ID test data')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--resume-training', type=bool, default=False, help='whether to resume training from a saved model')
    parser.add_argument('--save-model-dir', type=str, default='model/CNN', help='directory where trained model will be saved')
    parser.add_argument('--load-model-dir', type=str, default=None, help='path to trained model to resume training from')
    args = parser.parse_args()

    configure_logging()
    x_train, x_test, y_train, y_test = load_data(args.train_x_data_dir, args.test_x_data_dir, args.train_y_data_dir, args.test_y_data_dir)
    model = define_model(x_train, args.resume_training, args.load_model_dir)
    log_dir = os.path.join('../logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
    save_model_dir = args.save_model_dir
    train_model(model, x_train, y_train, x_test, y_test, args.epochs, args.batch_size, log_dir, save_model_dir)

if __name__ == '__main__':
    main()






