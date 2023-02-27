import os
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
from tqdm import tqdm
import librosa
import torch
import fairseq
import textgrid
from tensorflow.keras.utils import to_categorical
import argparse


#x
def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx
def melSpectra(y, sr, wsize, hsize):  #sound:(132300,) , sr=48000, wsize=0.04, hsize =0.04
    cnst = 1 + (int(sr * wsize) / 2) #883
    y_stft_abs = np.abs(librosa.stft(y, #132300
                                     win_length=int(sr * wsize),
                                     hop_length=int(sr * hsize),
                                     n_fft=int(sr * wsize))) / cnst
    melspec = np.log(1e-16 + librosa.feature.melspectrogram(sr=sr,
                                                            S=y_stft_abs ** 2,
                                                            n_mels=64))
    return melspec
def logmel_extractor(audio_dir, logmel_dir, num_frames, wsize, hsize, sr, n_steps):
    """
    Extract log-mel features from audio files in the given directory, and save the features to the given directory.

    Parameters:
        audio_dir (str): Path to directory containing audio files.
        logmel_dir (str): Path to directory where log-mel features will be saved.
        num_frames (int): Number of frames to use for log-mel features.
        wsize (float): Window size for log-mel feature extraction.
        hsize (float): Hop size for log-mel feature extraction.
        sr (int): Sampling rate for audio files.
        n_steps (int): Number of context steps for log-mel feature extraction.

    Returns:
        None
    """
    if not os.path.exists(logmel_dir):
        os.makedirs(logmel_dir)
    audio_subs = os.listdir(audio_dir)
    audio_subs.sort()
    if os.path.exists(audio_dir + '/.DS_Store') is True:
        audio_subs.remove('.DS_Store')
    sub_name_list = []
    c_sub = 0
    for a_audio_sub in audio_subs:  #33
        sub_name_list.append(a_audio_sub)
        c_sub += 1
        a_audio_sub_dir = audio_dir + '/' + a_audio_sub + '/'
        audio_seqs = os.listdir(a_audio_sub_dir) #1000
        audio_seqs.sort()
        if os.path.exists(a_audio_sub_dir + '.DS_Store') is True:
            audio_seqs.remove('.DS_Store')
            if len(audio_seqs) != 1000:
                print('num seqs in'+ a_audio_sub + 'is not 1000')


        cur_features_to_save = []
        for filename in audio_seqs:
            # Used for padding zeros to first and second temporal differences
            zeroVecD = np.zeros((1, 64), dtype='float32')
            zeroVecDD = np.zeros((2, 64), dtype='float32')

            # Load speech and extract features
            sound, sr = librosa.load(audio_dir + '/' + a_audio_sub + '/' + filename, sr=sr) #(132300,) sr=50000
            melFrames = np.transpose(melSpectra(sound, sr, wsize, hsize))
            melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
            melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
            features = np.concatenate((melDelta, melDDelta), axis=1)
            features = addContext(features, n_steps-1)
            features = np.reshape(features, (1, features.shape[0], features.shape[1]))
            upper_limit = features.shape[1]
            lower = 0

            for i in range(upper_limit):
                cur_features = np.zeros((1, num_frames, features.shape[2]))
                if i+1 > 75:
                    lower = i+1-75
                cur_features[:,-i-1:,:] = features[:,lower:i+1,:]
            cur_features.resize(np.shape(cur_features)[1],np.shape(cur_features)[2])
            cur_features_to_save.append(cur_features)
        if not os.path.exists(logmel_dir + '/' + a_audio_sub + '/'):
            os.makedirs(logmel_dir + '/' + a_audio_sub + '/')
        np.save(logmel_dir + '/'+ a_audio_sub + '/' + a_audio_sub + '.npy', cur_features_to_save)
    print('Log-Mel features extracted.')
def wav2vec_extractor(audio_dir, w2v_dir, w2v_model_dir, sr):
    """
    Extracts wav2vec C and Z features from audio clips using a pre-trained wav2vec model.

    Parameters:
        audio_dir (str): Path to directory containing audio files.
        w2v_dir (str): Path to output directory to save extracted wav2vec C or Z features.
        w2v_model_dir (str): Path to pre-trained wav2vec model directory.
        sr (int): Sampling rate of audio clips.

    Returns:
        None
    """
    # define 2 output directories for c and z
    w2v_c_dir = w2v_dir + '_c'
    w2v_z_dir = w2v_dir + '_z'

    # Load the feature extraction model
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([w2v_model_dir])
    model = model[0]
    model.eval()

    # Create the output directories if they don't exist
    if not os.path.exists(w2v_c_dir):
        os.makedirs(w2v_c_dir)
    if not os.path.exists(w2v_z_dir):
        os.makedirs(w2v_z_dir)

    # Get a list of the subdirectories in the audio directory
    audio_subs = os.listdir(audio_dir)
    audio_subs.sort()

    # Remove any non-directory files in the audio directory
    if os.path.exists(os.path.join(audio_dir, '.DS_Store')):
        audio_subs.remove('.DS_Store')

    # Loop over each subdirectory in the audio directory
    sub_name_list = []
    for a_audio_sub in tqdm(audio_subs):
        # Get the name of the subdirectory and the path to its directory
        sub_name_list.append(a_audio_sub)
        a_audio_sub_dir = os.path.join(audio_dir, a_audio_sub)

        # Get a list of the audio clips in the subdirectory
        audio_seqs = os.listdir(a_audio_sub_dir)
        audio_seqs.sort()

        # Remove any non-audio clip files in the subdirectory
        if os.path.exists(os.path.join(a_audio_sub_dir, '.DS_Store')):
            audio_seqs.remove('.DS_Store')

        # Check if the number of audio clips in the subdirectory is 1000
        if len(audio_seqs) != 1000:
            print('num seqs in' + a_audio_sub + 'is not 1000')

        # Loop over each audio clip in the subdirectory
        cur_c_features_to_save = []
        cur_z_features_to_save = []
        for filename in tqdm(audio_seqs):
            # Load the audio clip
            wav_input, sr = librosa.load(os.path.join(a_audio_sub_dir, filename), sr=sr)

            # Convert the audio clip to a PyTorch tensor and extract its features using the model
            wav_input = torch.from_numpy(wav_input).unsqueeze(0)
            z = model.feature_extractor(wav_input)
            z = z.detach().numpy()
            c = model.feature_aggregator(torch.from_numpy(z))
            c = c.detach().numpy()

            # Append the extracted features to the list of features to save
            cur_c_features_to_save.append(c)
            cur_z_features_to_save.append(z)

        cur_c_features_to_save = np.squeeze(cur_c_features_to_save, axis=1)
        cur_c_features_to_save = np.transpose(cur_c_features_to_save, (0, 2, 1))
        cur_z_features_to_save = np.squeeze(cur_z_features_to_save, axis=1)
        cur_z_features_to_save = np.transpose(cur_z_features_to_save, (0, 2, 1))

        # Create the subdirectories in the output directories if they don't exist
        if not os.path.exists(os.path.join(w2v_c_dir, a_audio_sub)):
            os.makedirs(os.path.join(w2v_c_dir, a_audio_sub))
        if not os.path.exists(os.path.join(w2v_z_dir, a_audio_sub)):
            os.makedirs(os.path.join(w2v_z_dir, a_audio_sub))
        # Save the extracted features as a NumPy array
        np.save(os.path.join(w2v_c_dir, a_audio_sub, a_audio_sub + '.npy'), cur_c_features_to_save)
        np.save(os.path.join(w2v_z_dir, a_audio_sub, a_audio_sub + '.npy'), cur_z_features_to_save)
def concatenate_features(data_dir):
    """
        Concatenates all NumPy arrays of the specified feature type within the specified directory, along the first axis,
        to generate a 3D array.
        Parameters:
            data_dir (str): The path to the directory containing feature NumPy arrays.
        Returns:
            ndarray: A 3D array (n_clip, n_Frame, n_feature).
    """
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                file_list.append(os.path.join(root, file))

    arrays = []
    for file_path in file_list:
        array = np.load(file_path)
        arrays.append(array)

    concatenated_array = np.concatenate(arrays, axis=0)
    return concatenated_array
def split_x_data(data_dir):
    """
        Splits the audio feature data into training and validation sets and saves them to .h5 files.

        Args:
            data_dir (str): Directory path containing log-Mel spectrogram or wav2vec feature data.
        """
    # Load the data
    file_name = os.path.basename(data_dir)
    data = concatenate_features(data_dir)

    # Get the number of samples
    n_samples = data.shape[0]

    # Shuffle the data
    np.random.seed(10)
    shuffled_data = np.random.permutation(data)

    # Divide the data into training and validation sets
    train_size = int(n_samples * 0.8)
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]

    # Save the training and validation sets
    train_name = os.path.join('data', file_name + '_train_data.h5')
    val_name = os.path.join('data', file_name + '_test_data.h5')
    with h5py.File(train_name, 'w') as f:
        f.create_dataset('data', data=train_data)
    with h5py.File(val_name, 'w') as f:
        f.create_dataset('data', data=val_data)
    return train_data, val_data


#y
def phoneme2viseme(phoneme_folder_path, num_frames, clip_time):
    """
        Extract visemes from TextGrid files and save them to a NumPy file.

        Args:
            phoneme_folder_path (str): Path to the folder containing phoneme TextGrid files.
            num_frames (int): Number of frames to use for viseme extraction. Defaults to 75.
            clip_time (int): Length of audio clip in seconds. Defaults to 3.

        Returns:
            ndarray: A 3D array (n_clip, n_Frame, n_feature).
        """
    print('Extracting visemes...')
    fps = num_frames/clip_time
    frames = np.arange(0, num_frames)
    seconds_per_frame = frames/fps
    phoneme_dict = {
        'ER0':9, 'AA1': 9, 'AE1': 9, 'AE2': 9, 'AH0': 9, 'AH1': 9, 'AO1': 11, 'AW1': 9, 'AY1': 9, 'B': 0, 'SH': 6, 'CH': 6, 'D': 5, 'DH': 3,
        'EH1': 9, 'EY1': 9, 'IY2': 9,'F': 2, 'G': 7, 'HH': 8, 'IH0': 9, 'IH1': 9, 'IY0': 9,'IY1': 9, 'NG':9, 'JH': 6, 'K': 7, 'L': 4,
        'M': 0, 'N': 5, 'OW0': 11, 'OW1': 11, 'OW2': 11, 'P': 0, 'R': 5, 'S': 6, 'T': 5, 'TH': 3, 'UW1': 10, 'V': 2, 'W': 1,
        'Y': 7, 'Z': 5, 'sil': 12, 'sp': 12
    }

    all_grid_dir = phoneme_folder_path
    grid_subs = os.listdir(all_grid_dir)
    grid_subs.sort()
    if os.path.exists(all_grid_dir + '.DS_Store') is True:
        grid_subs.remove('.DS_Store')

    viseme = []
    for sub in grid_subs:
        sub_grid_dir = os.path.join(all_grid_dir, sub)
        grid_seqs = os.listdir(sub_grid_dir)
        grid_seqs.sort()
        if os.path.exists(sub_grid_dir + '.DS_Store') is True:
            grid_seqs.remove('.DS_Store')

        all_frame_label = []
        for n in range(len(grid_seqs)):
            # Read a TextGrid object from a file.
            tg = textgrid.TextGrid.fromFile(os.path.join(sub_grid_dir, grid_seqs[n]))
            for t in seconds_per_frame:
                for i in range(len(tg[1][:])):
                    if tg[1][i].minTime <= t <tg[1][i].maxTime:
                        all_frame_label.append(phoneme_dict[tg[1][i].mark])
        viseme.append(all_frame_label)
    viseme = np.array(viseme)
    viseme = viseme.reshape(-1, num_frames)
    viseme_categorical = to_categorical(viseme)
    #viseme_numpy_file_name = os.path.join('data', 'visemeID.npy')
    #np.save(viseme_numpy_file_name, viseme)
    print('Viseme extracted.')
    return viseme_categorical
def split_y_data(data_dir, num_frames, clip_time):
    """
        Splits the viseme data into training and validation sets and saves them to .h5 files.

        Args:
            data_dir (str): Directory path containing the phoneme data from MFA.
            num_frames (int): Number of frames to use for viseme extraction.
            clip_time (int): Length of audio clip in seconds.
        """
    # Load the data
    data = phoneme2viseme(data_dir, num_frames, clip_time)

    # Get the number of samples
    n_samples = data.shape[0]

    # Shuffle the data
    np.random.seed(10)
    shuffled_data = np.random.permutation(data)

    # Divide the data into training and validation sets
    train_size = int(n_samples * 0.8)
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]

    # Save the training and validation sets
    train_name = os.path.join('data', str(num_frames) + '_viseme_train_data.h5')
    val_name = os.path.join('data', str(num_frames) + '_viseme_test_data.h5')
    with h5py.File(train_name, 'w') as f:
        f.create_dataset('data', data=train_data)
    with h5py.File(val_name, 'w') as f:
        f.create_dataset('data', data=val_data)
    return train_data, val_data








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run functions for processing data.')
    parser.add_argument('function', type=str, choices=['logmel_extractor', 'wav2vec_extractor', 'split_x_data', 'split_y_data'], help='Name of the function to run')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='path to directory of input, e.g. audio_dir, logmel_dir, w2v_c_dir')
    parser.add_argument('-o', '--output_dir', type=str, help='path to output directory to save extracted features or processed data')
    parser.add_argument('-n', '--num_frames', type=int, default=75, help='number of frames to use for audio features  (default=75)')
    parser.add_argument('--wsize', type=float, default=0.04, help='window size for log-mel feature extraction')
    parser.add_argument('--hsize', type=float, default=0.04, help='hop size for log-mel feature extraction')
    parser.add_argument('--sr', type=int, default=50000, help='sampling rate for audio files')
    parser.add_argument('--clip_time', type=int, default=3, help='length of audio clip in seconds')
    parser.add_argument('--n_steps', type=int, default=6, help='number of context steps for log-mel feature extraction')
    parser.add_argument('--w2v_model_dir', type=str, default='model/wav2vec_large.pt', help='path to directory containing the W2V model')
    args = parser.parse_args()

    if args.function == 'logmel_extractor':
        logmel_extractor(args.input_dir, args.output_dir, args.num_frames, args.wsize, args.hsize, args.sr, args.n_steps)
    elif args.function == 'wav2vec_extractor':
        wav2vec_extractor(args.input_dir, args.output_dir, args.w2v_model_dir, args.sr)
    elif args.function == 'split_x_data':
        split_x_data(args.input_dir)
    elif args.function == 'split_y_data':
        split_y_data(args.input_dir, args.num_frames, args.clip_time)



