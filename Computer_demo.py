import io
import os
import matplotlib.pyplot as plt
import numpy as np
import peakutils
import pyaudio
import wave
import time

from array import array
from scipy.fftpack import fft
from scipy.io import wavfile                                                    # get the api
from sys import byteorder
from struct import pack

np.set_printoptions(threshold=np.inf)

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

THRESHOLD = 1200
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
ARMED = 1

def speech_text(filename,samplerate):
    # Instantiates a client
    client = speech.SpeechClient()

    # The name of the audio file to transcribe
    file_name = ('%s.wav' %filename)
        

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=samplerate,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    for result in response.results:
        print('Transcript: {}'.format(result.alternatives[0].transcript))
        word = '{}'.format(result.alternatives[0].transcript)
        return word
    
    

def plotfft(filename):
    fs, a = wavfile.read('%s.wav' %filename)                                    # load the data
    b=[(ele/2**8.)*2-1 for ele in a]                                            # this is 8-bit track, b is now normalized on [-1,1)

    c = fft(b)                                                                  # calculate fourier transform (complex numbers list)
    d = int(len(c)/2)                                                           # you only need half of the fft list (real signal symmetry)
    e = abs(c[:(d-1)])                                                          #amplitudes

    #calculating frequency domain x-labels
    k = np.arange(len(a))
    T = len(a)/fs                                                               # where fs is the sampling frequency
    frqLabel = k/T
    xlabel = frqLabel[:(d-1)]

    #calculating time domain x-labels
    Ts = 1/fs
    Tlabel = np.linspace(0,Ts*len(b),len(b))

    fig, ax = plt.subplots(1,2)
    ax[0].plot(xlabel,e,'r')
    ax[1].plot(Tlabel,a,'r')

    plt.savefig('%s.png' %filename)
    #plt.show()
    return e, xlabel

def find_peaks(xlabel,amp):

    peaks = peakutils.indexes(amp, thres=0.3, min_dist=25)       #find peaks above threshold returns index
    pk2amp = [0]
    ind = [0]
    for j, pk in enumerate(peaks):                                              #sort peaks from highest to lowest amplitude response 
        pkamp = amp[pk]                                        
        for i, point in enumerate(pk2amp):
            if pkamp > point:
                pk2amp.insert(i,pkamp)
                ind.insert(i,j)
                break
            elif i == (len(pk2amp)-1):
                pk2amp.append(pkamp)
                ind.insert(i,j)
    pk2amp = pk2amp[:-1]
    ind = ind[:-1]

    '''
    print(pk2amp)
    print(ind)                                                                  #Hence sort indeces to show frequencies at which highest to lowest response occurs 
    print(peaks[ind])
    print(xlabel[peaks[ind]])
    '''
    pkfreq = xlabel[peaks[ind]]
    keyfreq = pkfreq[:10]                                                       #Shows top 10 frequencies
    print(keyfreq)
    return keyfreq, pkfreq

def view_peaks(peaks1,peaks2):

    results = open('Resultdata.txt', 'w')                                       #Export data to result file
    results.write('Sample 1 data \n')
    results.write('%s \n' %peaks1)
    results.write('Sample 2 data \n')
    results.write('%s \n' %peaks2)

    results.close()

def open_sesame(keyfreq,samplefreq,User):
    print('done')
    fdrift = 30
    match = []
    for sample in samplefreq:
        for key in keyfreq:
            if (key-fdrift)<=sample<=(key+fdrift):
                match.append(1)
                break
    print(len(match))
    print(len(keyfreq))

    if len(match)>= len(keyfreq)*0.80:
        print('Hello %s' %User)
        return 1
    else:
        print('Nice try imposter')
        return 0

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.05)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == '__main__':
    while(1):
        try:
            if ARMED == 1:
                print('Led solid red')
            elif ARMED == 0:
                print('LED solid green')
            time.sleep(5)

        except KeyboardInterrupt:
            for i in range(0,1):
                print('Led flashing green')
                #print('Please state Username')
                #record_to_file('username.wav')
                #User = speech_text('username',RATE)
                User = 'Angus'
                if User is None:
                    print('No speech detected')
                    print('Going back to sleep')
                    break
                elif not os.path.exists(User):
                    print('No user registered as this')
                    print('Going back to sleep')
                    break
                print("Please speak a password into the microphone")
                record_to_file('sample.wav')
                password = speech_text('sample',RATE)
                if password is None:
                    print('No speech detected')
                    print('Going back to sleep')
                    break
                password = password.strip().replace(" ","_")
                if not os.path.exists('%s/%s.wav' %(User, password)):
                    print('Password not registered to this User')
                    print('Going back to sleep')
                    break
                amp1, xlabel1 = plotfft('%s/%s' %(User, password))                                                     #sample fft of Reference file
                amp2, xlabel2 = plotfft('sample')                                                                      #sample fft of Attempt sample
                keyfreq1, peaks1 = find_peaks(xlabel1, amp1)   
                keyfreq2, peaks2 = find_peaks(xlabel2, amp2)
                #view_peaks(peaks1,peaks2)                                                                             #prints txt file to interigate peak values 
                if open_sesame(keyfreq1,keyfreq2,User) == 1:
                    ARMED = not ARMED
                if ARMED == 1:
                    print('Led flashing red')
                    #time.sleep(5)
                elif ARMED == 0:
                    print('Led solid green')
                    #time.sleep(5)
               

        
        
        
        
        