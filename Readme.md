# Audio to Viseme IDs Prediction
## Overview

This project aims to develop a model that can predict Viseme IDs from audio inputs.
It contains two methods for predicting Viseme IDs, one by Log-Mel spectrogram feature and the other by Wav2Vec feature.
## Set-up
- Python 3.9.16
- Tensorflow 2.5.0
- The requirements (including tensorflow) can be installed using:<br>
``` pip install -r requirements.txt```<br>
- Install graphviz:<br>
```conda install graphviz```<br>


## At test time:
#### 1. Create and install required envs and packages according to environment and set-up sections.
#### 2. Download this repository to your local machine <br>
```git clone https://vigitlab.fe.hhi.de/git/CVGAudioVisemePrediction.git ```<br>
#### 3. Prepare test data 
- The input audio needs to be 44.1kHz, 16-bit, WAV format. Put the file to `data/test/s/test_audio.wav`
- Process audio to Log-Mel spectrogram feature<br> 
```python data_process.py logmel_extractor -i data/test -o data/test_logmel --sr 50000 --n_steps 6```<br>
- Download wav2vec model from [wav2vec_large](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt). Put it to `model/` <br>
- Process audio to Wav2Vec feature ```python data_process.py wav2vec_extractor -i data/test -o data/test_wav2vec --sr 16000 --w2v_model_dir model/wav2vec_large.pt```<br>
#### 4. Prepare trained model:<br>
Download the trained models from `\\hhi.de\abteilung\VIT\VIT-CVG-Ablage\personal\Wolfgang\CVGAudio2AnimationPrediction\model`, put files to `model/`.
#### 5. Run commands to test model 
- Test logmel_lstm model: ```python predict_label_for_test.py logmel_lstm_predict_phoneme --model_dir model/to/model_name.h5 --x_test_dir data/path/to/logmel_test.npy```
- Test logmel_tdlstm model: ```python predict_label_for_test.py logmel_tdlstm_predict_phoneme --model_dir model/to/model_name.h5 --x_test_dir data/path/to/logmel_test.npy --delay 5```
- Test logmel_cnn model: ```python predict_label_for_test.py logmel_cnn_predict_phoneme --model_dir model/to/model_name.h5 --x_test_dir data/path/to/logmel_test.npy```

- Test wav2vec_lstm model: ```python predict_label_for_test.py w2v_lstm_predict_phoneme --model_dir model/to/model_name.h5 --x_test_dir data/path/to/w2v_test.npy```
- Test wav2vec_tdlstm model: ```python predict_label_for_test.py w2v_tdlstm_predict_phoneme --model_dir model/to/model_name.h5 --x_test_dir data/path/to/w2v_test.npy --delay 5```
- Test wav2vec_cnn model: ```python predict_label_for_test.py w2v_cnn_predict_phoneme --model_dir model/to/model_name.h5 --x_test_dir data/path/to/w2v_test.npy```
- Test wav2vec_mlp model: ```python predict_label_for_test.py w2v_mlp_predict_phoneme --model_dir model/to/model_name.h5 --x_test_dir data/path/to/w2v_test.npy```





## Train
### Data used to train
- Download the raw audios and word alignments from [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/).Choose raw 50 kHz audio.
- Put all downloaded raw audios folders to `data/audio`.
- Put all downloaded word alignments folders to `data/word`
### Input data processing
#### Log-Mel spectrogram feature
Extract the Log-Mel spectrogram feature of the audio <br>
```python data_process.py logmel_extractor -i data/audio -o data/logmel --sr 50000 --n_steps 6```<br>
Split Log-Mel spectrogram training and validation data sets <br>
```python data_process.py split_x_data -i data/logmel```
#### Wav2Vec feature
Download wav2vec model from [wav2vec_large](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt). Put it to `model/` <br>
Extract the Wav2Vec C and Z feature of the audio <br>
```python data_process.py wav2vec_extractor -i data/audio -o data/wav2vec --sr 16000 --w2v_model_dir model/wav2vec_large.pt```<br>
Split C feature training and validation data sets <br>
```python data_process.py split_x_data -i data/wav2vec_c```<br>
Split Z feature training and validation data sets <br>
```python data_process.py split_x_data -i data/wav2vec_z```

### Viseme ID lable
#### Install the Montreal Forced Aligner:<br>
For installation instructions, please refer to the [MFA installation page](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html#windows). <br>
To install MFA, you will need to have Conda/Miniconda and Kaldi (depending on your system version) installed.<br>
1. Install Anaconda<br>
2. Switch to a new environment: <br>
```conda create -n aligner -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch```<br>
```conda activate aligner```<br>
3. Install MFA: <br>
```pip install montreal-forced-aligner```<br>
4. Install third-party packages:<br>
```mfa thirdparty download```<br>
5. Upgrade to the latest version:<br>
```pip install montreal-forced-aligner -U```<br>
6. Verify installation:<br>
```mfa thirdparty validate```<br>
After completing these steps, you can check the version of MFA by running:<br>
```mfa version```<br>
#### Preparation for Forced Alignment
To perform Forced Alignment, we need to prepare the following three things:
- Pronunciation dictionary for the language.
- Audio data (.wav)
- Text data. Note that the name of the text data file should correspond one-to-one with the corresponding audio data, except for the extension.
#### Steps for Forced Alignment <br>
1. Download the English acoustic model:<br>
```mfa download acoustic english```<br>
2. Download the English pronunciation dictionary:<br>
There are many options available, but a commonly used one is [LibriSpeech Lexicon](https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/english.dict).<br>
3. Perform Forced Alignment:<br>
Use the pre-trained model to perform alignment:<br>
```mfa align /path/to/dataset /path/to/lexicon.txt english /output/path```<br>
#### Process the MFA output
1. Arrange the .textgrid files generated by MFA in the `phoneme/s1/.textgrid` format used by the GRID database. 
2. Process .textgrid files and split viseme ID training and validation data sets for Log-Mel spectrogram feature training<br>
```python data_process.py split_y_data -i data/phoneme -n 75  --clip_time 3```<br>
3. Process .textgrid files and split viseme ID training and validation data sets for Wav2Vec feature training<br>
```python data_process.py split_y_data -i data/phoneme -n 298  --clip_time 3```
### Training
#### Log-Mel spectrogram feature as input
Train LSTM model with default arguments<br>
```python train/train_logmel_lstm.py --epochs 300 --batch-size 32```<br>
Train TDLSM model with default arguments<br>
```python train/train_logmel_tdlstm.py --epochs 300 --batch-size 32```<br>
Train CNN model with default arguments<br>
```python train/train_logmel_cnn.py --epochs 300 --batch-size 32```<br>
#### Wav2Vec feature as input
Train LSTM model with default arguments<br>
```python train/train_w2v_lstm.py --epochs 300 --batch-size 32```<br>
Train TDLSM model with default arguments<br>
```python train/train_w2v_tdlstm.py --epochs 300 --batch-size 32```<br>
Train CNN model with default arguments<br>
```python train/train_w2v_cnn.py --epochs 300 --batch-size 32```<br>
Train MLP model with default arguments<br>
```python train/train_w2v_mlp.py --epochs 300 --batch-size 32```


