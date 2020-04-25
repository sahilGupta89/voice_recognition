#!/usr/bin/env python
#!/usr/local/bin/python
"""Utilities for downloading and providing data from openslr.org, libriSpeech, Pannous, Gutenberg, WMT, tokenizing, vocabularies."""
# TODO! see https://github.com/pannous/caffe-speech-recognition for some data sources

import os
import re
import sys
import wave

import numpy
import numpy as np
import skimage.io  # scikit-image
import tensorflow as tf

try:
    import librosa
except:
    print("pip install librosa ; if you want mfcc_batch_generator")
# import extensions as xx
from random import shuffle
try:
    from six.moves import urllib
    from six.moves import xrange  #
except:
    pass






# works as input folder path for mfcc generation
pcm_path = "/home/deeplearning/Desktop/learn/data/" # 8 bit
 # 16 bit s16le
path = pcm_path

#  important  input folder for  files to be predicted
test_path = "/home/deeplearning/Desktop/learn/test_data/"
CHUNK = 4096
test_fraction=0.1 # 10% of data for test / verification
outfile = 'D:/Neural_Network_Research/Database/dictionaries/dict_of_arrays.npz'

offset = 64  # starting with characters
max_word_length = 20
terminal_symbol = 0
num_characters = 32

def char_to_class(c):
    return (ord(c) - offset) % num_characters



from enum import Enum
class Target(Enum):  # labels
    digits=1
    speaker=2
    words_per_minute=3
    word_phonemes=4
    word = 5  # int vector as opposed to binary hotword
    sentence=6
    sentiment=7
    first_letter=8
    hotword = 9

def pad(vec, pad_to=max_word_length, one_hot=False,paddy=terminal_symbol):
    for i in range(0, pad_to - len(vec)):
        if one_hot:
            vec.append([paddy] * num_characters)
        else:
            vec.append(paddy)
    return vec


# important
def string_to_int_word(word, pad_to):
    z = map(char_to_class, word)
    z = list(z)
    z = pad(z)
    return z

class SparseLabels:
    def __init__(labels):
        labels.indices = {}
        labels.values = []

    def shape(self):
        return (len(self.indices),len(self.values))

# labels: An `int32` `SparseTensor`.
# labels.indices[i, :] == [b, t] means `labels.values[i]` stores the id for (batch b, time t).
# labels.values[i]` must take on values in `[0, num_labels)`.
def sparse_labels(vec):
    labels = SparseLabels()
    b=0
    for lab in vec:
        t=0
        for c in lab:
            labels.indices[b, t] = len(labels.values)
            labels.values.append(char_to_class(c))
            # labels.values[i] = char_to_class(c)
            t += 1
        b += 1
    return labels








def speaker(filename):  # vom Dateinamen
    # if not "_" in file:
    #   return "Unknown"
    return filename.split("_")[1]


# important
def get_speakers(path=pcm_path):
    path = path
    files = os.listdir(path)
    def nobad(name):
        name.replace('.','-')
        # print(name)
        return "_" in name and not "." in name.split("_")[0]
    speakers=list(set(map(speaker,filter(nobad,files))))
    # print(speakers)
    # print(len(speakers)," speakers: ",speakers)
    return speakers,len(files)





# important converts audio files to features and saves dictionary containing mfcc's

def mfcc_batch_generator(batch_size=10, target=Target.digits,start=0,path=pcm_path,out_file =outfile):

    if target == Target.speaker: speakers = get_speakers(path)
    batch_features = []
    labels = []
    files = os.listdir(path)
    # shuffle(files)
    files_to_process = files[start:batch_size+start]

    dict_file ={}
    # dict_file ={}

    if os.path.exists(out_file):
        print("mfcc found")
        while True:
            npzfile = np.load(out_file)
            print('npzfile.files: {}'.format(npzfile.files))
            for file in files_to_process:
                if not file.endswith(".wav"): continue
                try:
                    batch_features.append(npzfile[file])
                    labels.append(one_hot_from_item(speaker(file),speakers[0]))
                except Exception as ex:
                    wave, sr = librosa.load(path + file, mono=True)
                    mfcc = librosa.feature.mfcc(wave, sr)
                    mfcc = np.pad(mfcc, ((0, 0), (0, 1077 - len(mfcc[0]))), mode='constant', constant_values=0)
                    mfcc.flatten()

                    # dict_file=npzfile.files
                    dict_file = {key:npzfile[key] for key in npzfile.files}
                    dict_file[file] = np.array(mfcc)
                    batch_features.append(np.array(mfcc))
                    labels.append(one_hot_from_item(speaker(file),speakers[0]))




                if len(batch_features) >= batch_size:
                    if len(dict_file)>0:
                        # outfile = 'dict_of_arrays.npz'
                        np.savez(out_file, **dict_file)


                    yield  batch_features, labels
                    batch_features = []  # Reset for next batch
                    labels = []

    else:


        while True:
            print("loaded batch of %d files" % len(files))
            shuffle(files_to_process)
            for file in files_to_process:
                if not file.endswith(".wav"): continue
                wave, sr = librosa.load(path+file, mono=True)
                mfcc = librosa.feature.mfcc(wave, sr)
                if target==Target.speaker: label=one_hot_from_item(speaker(file), speakers[0])
                elif target==Target.digits:  label=dense_to_one_hot(int(file[0]),10)
                elif target==Target.first_letter:  label=dense_to_one_hot((ord(file[0]) - 48) % 32,32)
                elif target == Target.hotword: label = one_hot_word(file, pad_to=max_word_length)  #
                elif target == Target.word: label=string_to_int_word(file, pad_to=max_word_length)
            # label = file  # sparse_labels(file, pad_to=20)  # max_output_length
                else: raise Exception("todo : labels for Target!")
                labels.append(label)

            # print(np.array(mfcc).shape)
                mfcc=np.pad(mfcc,((0,0),(0,1077-len(mfcc[0]))), mode='constant', constant_values=0)
                mfcc.flatten()
                hot = np.argmax(label)
                hot = speakers[0][hot]
                dict_file[file] =np.array(mfcc)
                batch_features.append(np.array(mfcc))
                if len(batch_features) >= batch_size:
                # if target == Target.word:  labels = sparse_labels(labels)
                # labels=np.array(labels)
                # print(np.array(batch_features).shape)
                # yield np.array(batch_features), labels
                # print(np.array(labels).shape) # why (64,) instead of (64, 15, 32)? OK IFF dim_1==const (20)

                    np.savez(out_file, **dict_file)

                    yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                    batch_features = []  # Reset for next batch
                    labels = []







# important converts result to probabilities and labels
def prob_to_result(hot, items):
    list_result=[]
    dict_result={}
    hot = np.asarray(hot)
    for i in range(len(hot[0])):
        if hot[0][i]>0.0001:
            dict_result = {}

            dict_result["Label"]=items[i]
            dict_result["Percentage"]=hot[0][i]*100
            list_result.append(dict_result)


    # i=np.argmax(hot)
    # item=items[i]
    return list_result
# important converts one hot to item
def one_hot_from_item(item, items):
    # items=set(items) # assure uniqueness
    x=[0]*len(items)# numpy.zeros(len(items))
    i=items.index(item)
    x[i]=1
    return x

# important
def one_hot_word(word,pad_to=max_word_length):
    vec=[]
    for c in word:#.upper():
        x = [0] * num_characters
        x[(ord(c) - offset)%num_characters]=1
        vec.append(x)
    if pad_to:vec=pad(vec, pad_to, one_hot=True)
    return vec



# important
def dense_to_one_hot(batch, batch_size, num_labels):
    sparse_labels = tf.reshape(batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    concatenated = tf.concat(axis=1, values=[indices, sparse_labels])
    concat = tf.concat(axis=0, values=[[batch_size], [num_labels]])
    output_shape = tf.reshape(concat, [2])
    sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
    return tf.reshape(sparse_to_dense, [batch_size, num_labels])




