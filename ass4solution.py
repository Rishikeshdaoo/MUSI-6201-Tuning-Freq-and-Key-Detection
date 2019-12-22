import os
import glob
import numpy as np
import matplotlib
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
from scipy.signal import find_peaks
import scipy as sp
import math

def block_audio(x, blockSize, hopSize, fs):
    """
    Sample audio blocking code from Alex
    """
    # allocate memory
    numBlocks = int(np.ceil(x.size / hopSize))
    xb = np.zeros([numBlocks, blockSize])

    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return xb, t


def compute_hann(iWindowLength):
    """
    Sample compute hann window code from Alex
    """
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))


def readwav(filename):
    fs, x = wavread(filename)
    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32
        else:
            nbits = 1  # No conversion

        audio = x / float(2 ** (nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    if audio.ndim > 1:
        audio = audio[:, 0]

    return fs, audio

def compute_spectrogram(xb,fs):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(sp.fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) #let's be pedantic about normalization

    f = np.arange(0, X.shape[0])*fs/(xb.shape[1])
    
    return (X,f)


#A. Tuning Frequency Estimation: [25points]
"""
1. [5 points] Write a function get_spectral_peaks(X) that returns the top 20 spectral peak bins of each column of magnitude spectrogram X.
"""
def get_spectral_peaks(X: np.ndarray) -> np.ndarray:
    C = X.shape[1]
    out = np.zeros((20, C), dtype=np.int16)
    for column in range(X.shape[1]):
        spectrogram = X[:,column]
        top20_idx = spectrogram.argsort(axis = 0)[-20:][::-1]
        
        # out[:,column] = spectrogram[top20_idx]
        out[:,column] = top20_idx
    return out


"""
2. [20 points] Write a function estimate_tuning_freq(x, blockSize, hopSize, fs) to return the tuning frequency. 
x is the time domain audio signal, blockSize is the block size, hopSize is the hopSize, and fs is the sample rate. 
Use the deviation from the equally tempered scale in Cent for your estimate. 
You will use get_spectral_peaks() function to obtain the top 20 spectral peaks for each block. 
Use functions from the reference solutions for previous assignments for blocking and computing the spectrogram. 
For each block, compute the deviation of the peak bins from the nearest equally tempered pitch bin in cents.
Then, compute a histogram of the deviations in Cent and derive the tuning frequency from the location of the max of 
this histogram.
"""


def estimate_tuning_freq(x, blockSize, hopSize, fs):
       
    pitch = np.array([440 * (2 ** ((p-69)/12)) for p in range(128)])

    estimate_freq = None

    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    spec, freq = compute_spectrogram(xb, fs)
    peak_idx = get_spectral_peaks(spec)

    max_freqs = freq[peak_idx]
    # print(max_freqs)
    assert max_freqs.shape == peak_idx.shape

    min_error =  np.zeros(peak_idx.shape)
    for i in range(peak_idx.shape[0]):
        for j in range(peak_idx.shape[1]):
            if max_freqs[i][j] != 0:
                
                min_error[i][j] = np.min(np.abs(1200* np.log2(max_freqs[i][j]/pitch)))
    
    # cent bin resolution: 10 cents
    bin_nums = int((np.round(np.max(min_error)) - np.round(np.min(min_error))) // 10)
    # print(bin_nums)
    hist, bin_edges = np.histogram(min_error,bins=bin_nums)
    
    diff = bin_edges[np.argmax(hist)]
    estimate_freq = 440 * np.power(2, diff/1200)

    return estimate_freq


"""
#B. Key Detection: [50 points]
1. [25 points] Write a function extract_pitch_chroma(X, fs, tfInHz) which returns the pitch chroma array (dimensions 12 x numBlocks). 
X is the magnitude spectrogram, fs is the sample rate, and tfInHz is the tuning frequency. Compute the pitch chroma for the 3 octave 
range C3 to B5. You will need to adjust the semitone bands based on your calculation of tuning frequency deviation. Each individual 
chroma vector should be normalized to a length of 1.
"""


def extract_pitch_chroma(X, fs, tfInHz):
    block_length = X.shape[0]
    num_blocks = X.shape[1]
    pitch_chroma = np.zeros((12, block_length), dtype=np.int16)

    Y = np.abs(X) ** 2

    # Need to calculate pitch chroma from C3 to B5 --> 48 to 83
    lower_bound = 48
    upper_bound = 84

    k = np.arange(1, (block_length+1))
    k_freq = k * fs / (2 * (block_length-1))

    irange = (upper_bound-lower_bound)

    logfreq_X = np.zeros([irange, num_blocks])

    for n, i in enumerate(range(lower_bound, upper_bound)):

        midi_pitch_lower = 2 ** (((i - 0.5) - 69) / 12) * tfInHz
        midi_pitch_upper = 2 ** (((i + 0.5) - 69) / 12) * tfInHz

        mask = np.logical_and(midi_pitch_lower <= k_freq, k_freq < midi_pitch_upper)

        logfreq_X[n, :] = Y[k[mask], :].sum(axis=0)


    pitch_chroma = np.zeros((12, logfreq_X.shape[1]))
    p = np.arange(48,84)
    for c in range(12):
        mask = (p % 12) == c
        pitch_chroma[c, :] = logfreq_X[mask, :].sum(axis=0)

    idx = [9,10,11,0,1,2,3,4,5,6,7,8]

    pitch_chroma = pitch_chroma[idx, :]

    # normalize to unit length
    l2norm = np.linalg.norm(pitch_chroma, ord=2, axis=0)
    l2norm[l2norm == 0] = 1
    pitch_chroma /= l2norm
    
    return pitch_chroma

"""
2. [25 points] Write a function detect_key(x, blockSize, hopSize, fs, bTune) to detect the key of a given audio signal. 
The parameter bTune is True or False and will specify if tuning frequency correction is done for computing the key. 
The template profiles to use for estimating the key are the Krumhansl templates:
"""


def detect_key(x:np.ndarray, blockSize:int, hopSize:int, fs: int, bTune: bool) -> int:
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])

    # normalize template #
    t_pc = t_pc / np.linalg.norm(t_pc, ord = 2, axis = 1,keepdims= True)
    pred_key = None
    
    # if tuned
    if bTune:
        tfInHz = estimate_tuning_freq(x,blockSize,hopSize,fs)
    # if not tuned
    else:
        tfInHz = 440
    xb, _ = block_audio(x, blockSize, hopSize, fs)
    X, _ = compute_spectrogram(xb, fs)

    pitch_chroma = extract_pitch_chroma(X, fs, tfInHz)
    mean_pc = np.mean(pitch_chroma, axis = 1)

    distance = np.zeros((24,))

    for i in range(12):
        distance[i] = np.linalg.norm(mean_pc - np.roll(t_pc[0],i), ord = 2)
    for i in range(12):
        distance[12+i] = np.linalg.norm(mean_pc - np.roll(t_pc[1],i), ord = 2)
    pred_key = np.argmin(distance)
    return pred_key

#Note that this array contains both major and minor key profiles, respectively. Hint: don't forget to normalize the key 
#profiles. Use Euclidean distance.


#########################################################
#           Helper Functions for Evaluation             #
#########################################################
def read_dir(root_directory:str):
    file_array = []
    for dirpath, _, filenames in os.walk(root_directory):
        # print(filenames)
        for filepath in sorted(filenames):
            sub = os.path.splitext(filepath)[1]
            if(sub == ".txt" or sub == ".wav"):
                relative_path = os.path.join(dirpath,filepath)
                file_array.append(relative_path)
    return file_array

def read_gt(file_dir):
    with open(file_dir, "r") as f:
        return f.readline() 

def freq_diff_cents(estimate_freq:float, target_freq:float) -> float:
    """
    Compute the absolute difference between the estimate frequency and the target frequency in cents
    
    Inputs:
        estimate_freq: estimated frequency in Hz
        target_freq: float, ground truth frequency in Hz

    Returns:
        abs_diff_in_cents: float, absolute difference in cents
    """

    assert(target_freq !=0) , "target freqency equals to 0"
    abs_diff_in_cents = 1200 * np.abs(np.log2(estimate_freq/target_freq))

    return abs_diff_in_cents

"""
C. Evaluation: [25 points]
For the evaluation use blockSize = 4096, hopSize = 2048.

#1. [10 points] Write a function eval_tfe(pathToAudio, pathToGT) that evaluates tuning frequency estimation for the 
audio files in the folder pathToAudio. For each file in the audio directory, there will be a corresponding .txt file 
in the GT directory containing the ground truth tuning frequency deviation in Cent. You return the average absolute 
deviation of your tuning frequency estimation in cents for all the files.
"""

def eval_tfe(pathToAudio, pathToGT):

    blockSize = 4096
    hopSize = 2048

    audio_files = read_dir(pathToAudio)
    ground_truths = read_dir(pathToGT)

    assert len(audio_files) == len(ground_truths), "The numbers of audio files and ground truth files do not match!"
    assert (len(audio_files)) != 0, "No file detected. Must include at least one file"
    
    total_error = 0
    N = len(audio_files)
    for (audio, gt) in zip(audio_files,ground_truths):

        # make sure audio file name matches gt file name
        audio_name = os.path.splitext(os.path.split(audio)[1])[0]
        gt_name = os.path.splitext(os.path.split(gt)[1])[0]

        assert  audio_name == gt_name , "Audio file names %s and gt file name %s does not match" % (audio_name, gt_name)

        fs, x = readwav(audio)
        estimate_freq = estimate_tuning_freq(x, blockSize, hopSize, fs)
        target_freq = float(read_gt(gt))

        diff = freq_diff_cents(estimate_freq, target_freq)
        
        total_error += diff

    return total_error/N   

        
"""
2. [10 points] Write a function eval_key_detection(pathToAudio, pathToGT) that evaluates key detection for the audio 
files in pathToAudio. For each file in the audio directory, there will be a corresponding .txt file in the GT directory 
containing the ground truth key label. 
You return the accuracy = (number of correct key detections) / (total number of songs) of your key detection for all 
the files with and without tuning frequency estimation.The output accuracy will be an np.array dimensions 2 x 1 vector 
with the first element the accuracy with tuning frequency correction and the second without tuning frequency correction.
"""
def eval_key_detection(pathToAudio, pathToGT):
    blockSize = 4096
    hopSize = 2048

    audio_files = read_dir(pathToAudio)
    ground_truths = read_dir(pathToGT)
    assert len(audio_files) == len(ground_truths), "The numbers of audio files and ground truth files do not match!"
    assert (len(audio_files)) != 0, "No file detected. Must include at least one file"

    total_accuracy = np.zeros((2,))
    N = len(audio_files)

    for (audio, gt) in zip(audio_files,ground_truths):

        # make sure audio file name matches gt file name
        audio_name = os.path.splitext(os.path.split(audio)[1])[0]
        gt_name = os.path.splitext(os.path.split(gt)[1])[0]
        assert  audio_name == gt_name , "Audio file names %s and gt file name %s does not match" % (audio_name, gt_name)

        fs, x = readwav(audio)
        target_key = int(read_gt(gt))

        # with tuning
        pred_key_without_tuning = detect_key(x, blockSize, hopSize, fs, True)
        total_accuracy[0] += (pred_key_without_tuning == target_key)

        # without tuning
        pred_key_tuned = detect_key(x, blockSize, hopSize, fs, False)
        total_accuracy[1] += (pred_key_tuned == target_key)

    return total_accuracy/N   


"""
[5 points] Write a function evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf) which runs the above two 
functions with the data given in the respective directories (key_tf.zip). Report the average absolute deviation for 
the tuning frequency estimation in Cent and the accuracy for key detection with and without tuning frequency correction.
"""
def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):
    
    freq_estimate_error = eval_tfe(pathToAudioTf, pathToGTTf)
    key_detection_acc = eval_key_detection(pathToAudioKey, pathToGTKey)

    print("Frequency estimate error: %s, key detection accuracy: with tuning freq estimate: %s, without tuning freq estimate: %s" % (freq_estimate_error, key_detection_acc[0], key_detection_acc[1]))

if __name__ == "__main__":
    blockSize = 4096
    hopSize = 2048

    pathToAudioKey= "./key_tf/key_eval/audio/"
    pathToGTKey = "./key_tf/key_eval/GT/"
    pathToAudioTf = "./key_tf/tuning_eval/audio/"
    pathToGTTf = "./key_tf/tuning_eval/GT/"

    evaluate(pathToAudioKey, pathToGTKey, pathToAudioTf, pathToGTTf)
