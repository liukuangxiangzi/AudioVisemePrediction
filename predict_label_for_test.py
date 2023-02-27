import os
import argparse
import h5py
from keras.models import load_model
import numpy as np

# Define the dictionary
phoneme_dict = {
    'ER0': 9, 'AA1': 9, 'AE1': 9, 'AE2': 9, 'AH0': 9, 'AH1': 9, 'AO1': 11, 'AW1': 9, 'AY1': 9, 'B': 0, 'SH': 6,
    'CH': 6, 'D': 5, 'DH': 3,
    'EH1': 9, 'EY1': 9, 'IY2': 9, 'F': 2, 'G': 7, 'HH': 8, 'IH0': 9, 'IH1': 9, 'IY0': 9, 'IY1': 9, 'NG': 9, 'JH': 6,
    'K': 7, 'L': 4,
    'M': 0, 'N': 5, 'OW0': 11, 'OW1': 11, 'OW2': 11, 'P': 0, 'R': 5, 'S': 6, 'T': 5, 'TH': 3, 'UW1': 10, 'V': 2,
    'W': 1,
    'Y': 7, 'Z': 5, 'sil': 12, 'sp': 12
}

def logmel_lstm_predict_phoneme(model_path, x_test_path):
    # Load model
    model = load_model(model_path)

    # Load test data
    testx = np.load(x_test_path)
    testx = testx.reshape(testx.shape[0] * testx.shape[1], -1)
    testx = testx.reshape(-1, 6, 128)
    print('testx', testx.shape)
    y_pred = np.squeeze(model.predict(testx))
    phonemes = [list(phoneme_dict.keys())[list(phoneme_dict.values()).index(label)] for label in y_pred.argmax(axis=1)]
    print(phonemes)
    return y_pred
def logmel_tdlstm_predict_phoneme(model_path, x_test_path, delay):
    # Load model
    model = load_model(model_path)

    # Load test data
    testx = np.load(x_test_path)
    testx = testx.reshape(testx.shape[0] * testx.shape[1], -1)
    testx = testx.reshape(-1, 6, 128)
    testx = testx[delay:, :, :]

    print('testx', testx.shape)
    y_pred = np.squeeze(model.predict(testx))
    phonemes = [list(phoneme_dict.keys())[list(phoneme_dict.values()).index(label)] for label in y_pred.argmax(axis=1)]
    print(phonemes)
    return y_pred

def logmel_cnn_predict_phoneme(model_path, x_test_path):
    # Load model
    model = load_model(model_path)

    # Load test data
    testx = np.load(x_test_path)
    testx = testx.reshape(testx.shape[0] * testx.shape[1], -1)
    testx = testx.reshape(-1, 6, 128)

    print('testx', testx.shape)
    y_pred = np.squeeze(model.predict(testx))
    phonemes = [list(phoneme_dict.keys())[list(phoneme_dict.values()).index(label)] for label in y_pred.argmax(axis=1)]
    print(phonemes)
    return y_pred
def w2v_lstm_predict_phoneme(model_path, x_test_path):
    # Load model
    model = load_model(model_path)

    # Load test data
    testx = np.load(x_test_path)
    print('testx', testx.shape)
    y_pred = np.squeeze(model.predict(testx))
    print('y_pred', y_pred.shape)
    phonemes = [list(phoneme_dict.keys())[list(phoneme_dict.values()).index(label)] for label in y_pred.argmax(axis=1)]
    print(phonemes)
    return y_pred
def w2v_tdlstm_predict_phoneme(model_path, x_test_path, delay):
    # Load model
    model = load_model(model_path)

    # Load test data
    testx = np.load(x_test_path)
    testx = testx[:, delay:, :]
    print('testx', testx.shape)
    y_pred = np.squeeze(model.predict(testx))
    phonemes = [list(phoneme_dict.keys())[list(phoneme_dict.values()).index(label)] for label in y_pred.argmax(axis=1)]
    print(phonemes)
    return y_pred
def w2v_cnn_predict_phoneme(model_path, x_test_path):
    # Load model
    model = load_model(model_path)

    # Load test data
    testx = np.load(x_test_path)
    print('testx', testx.shape)
    y_pred = np.squeeze(model.predict(testx))
    phonemes = [list(phoneme_dict.keys())[list(phoneme_dict.values()).index(label)] for label in y_pred.argmax(axis=1)]
    print(phonemes)
    return y_pred
def w2v_mlp_predict_phoneme(model_path, x_test_path):
    # Load model
    model = load_model(model_path)

    # Load test data
    testx = np.load(x_test_path)
    print('testx', testx.shape)
    y_pred = np.squeeze(model.predict(testx))
    phonemes = [list(phoneme_dict.keys())[list(phoneme_dict.values()).index(label)] for label in y_pred.argmax(axis=1)]
    print(phonemes)
    return y_pred


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser(description='Predict phonemes using trained models.')
    parser.add_argument('function', type=str, choices=['logmel_lstm_predict_phoneme','logmel_lstm_predict_phoneme','logmel_cnn_predict_phoneme',
    'w2v_lstm_predict_phoneme','w2v_tdlstm_predict_phoneme','w2v_cnn_predict_phoneme','w2v_mlp_predict_phoneme'], help='Name of the function to run')
    parser.add_argument('--model_dir', type=str, help='path to the model file')
    parser.add_argument('--x_test_dir', type=str, help='path to the test data file')
    parser.add_argument('--delay', type=int, default=5, help='delay frame_num')
    args = parser.parse_args()

    if args.function == 'logmel_lstm_predict_phoneme':
        logmel_lstm_predict_phoneme(args.model_dir, args.x_test_dir)
    elif args.function == 'logmel_tdlstm_predict_phoneme':
        logmel_tdlstm_predict_phoneme(args.model_dir, args.x_test_dir, args.delay)
    elif args.function == 'logmel_cnn_predict_phoneme':
        logmel_cnn_predict_phoneme(args.model_dir, args.x_test_dir)


    elif args.function == 'w2v_lstm_predict_phoneme':
        w2v_lstm_predict_phoneme(args.model_dir, args.x_test_dir)
    elif args.function == 'w2v_tdlstm_predict_phoneme':
        w2v_tdlstm_predict_phoneme(args.model_dir, args.x_test_dir, args.delay)
    elif args.function == 'w2v_cnn_predict_phoneme':
        w2v_cnn_predict_phoneme(args.model_dir, args.x_test_dir)
    elif args.function == 'w2v_mlp_predict_phoneme':
        w2v_mlp_predict_phoneme(args.model_dir, args.x_test_dir)