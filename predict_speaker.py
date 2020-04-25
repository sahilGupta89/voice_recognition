#! C:/Users/mnj2/AppData/Local/conda/conda/envs/myenv/

# try:

import os

import tflearn


import preprocess_get_data as data

import librosa
# from python_speech_features import  mfcc
try:
    from python_speech_features import mfcc
    from python_speech_features import logfbank
    import scipy.io.wavfile as wav
except Exception as ex:
    print(ex)
    s = input("Done")

import numpy as np
import argparse
# except Exception as ex:
#     with open('D:/Neural_Network_Research/Resources/log.txt','w') as file:
#         file.writelines(str(ex))
# import tensorflow as tf
# print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
# if tf.__version__ >= '0.12' and os.name == 'nt':
#     print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
#     quit() # why? works on Mac?


INPUTFOLDERPATH =""
PREDICTIONFILESPATH=""
DICTIONARYPATH =""
SPEAKERSFILE =""
EPOCHSNO =0
BATCHSIZE=0
LEARNINGRATE=0.0
OPTIMIZER=""
ACTIVATION=""
LOOPVALUE=""
MODELPATH=""
LOSSFUNCTION=""
OUTPUTFILECSV=""

# argument parser

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Wav files should be in format 16-44k , mono channel  , 8 - bits  and length of each file not exceeding 25 secs

Examples:
    To be decided
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser = argparse.ArgumentParser(description=desc, epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--inputfolder',
                        help='works as input folder path for mfcc generation',

                        default='D:/Neural_Network_Research/Resources/Output_Data/')

    parser.add_argument('-pf', '--predictionfile',
                        help='Folder path where files to be predicted is stored',
                        default='D:/Neural_Network_Research/Resources/Output_Data/Test_data/')

    parser.add_argument('-d', '--dictionary',
                        help='Folder path for dictionary to be stored for fast calculation of mfcc ',
                        default="D:/Neural_Network_Research/Resources/Output_Data/dictionaries/dict_of_arrays.npz")
    parser.add_argument('-m', '--modelpath',
                        help='Folder path where model is to be stored',
                        default='D:/Neural_Network_Research/Resources/Output_Data/models/model.tfl')

    parser.add_argument('-sf', '--speakersfile',
                        help='File path for dictionary to be stored speakers list used while matching or predicting label',
                            default="D:/Neural_Network_Research/Resources/Output_Data/speaker_list.npy")
    parser.add_argument('-e', '--epochs',
                        help='No. of training epochs',
                        type=int,
                        default=600)
    parser.add_argument('-b', '--batchsize',
                        help='Size of batch used for training after getting all the mfccs',
                        type=int,
                        default=200)

    parser.add_argument('-l', '--learningrate',
                        help='Learing rate for neural network',
                        type=float,
                        default=0.00001)
    parser.add_argument('-o', '--optimizer',
                        help='Select appropriate optimizer for examlpe "adam"',
                        default="adam")

    parser.add_argument('-a', '--activation',
                        help='Activation function for neural network for example "sigmoid"',

                        default="sigmoid")
    parser.add_argument('-lv', '--loopvalue',
                        help='iteration over data for better accuracy default = 1',
                        type=int,
                        default=1)
    parser.add_argument('-lf', '--lossfunction',
                        help='define loss function for neural network',

                        default="categorical_crossentropy")
    parser.add_argument('-of', '--outputfile',
                        help='define loss function for neural network',

                        default="D:/Neural_Network_Research/Resources/Output_Data/output_speaker_classification.csv")


    ret = parser.parse_args()
    return ret


def predict_speaker():
    speakers,files_length = data.get_speakers(path=INPUTFOLDERPATH)

    number_classes=len(speakers)

    #    Classification
    tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

    # net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
    net = tflearn.input_data(shape=[None,20,1077]   ) #Two wave chunks


    net = tflearn.fully_connected(net, 64)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, number_classes, activation=ACTIVATION,)

    # net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
    net = tflearn.regression(net, optimizer=OPTIMIZER,learning_rate=LEARNINGRATE, loss=LOSSFUNCTION)
    # net = tflearn.
    model= tflearn.DNN(net)
    # sess = tf.Session()
    # tflearn.is_training(False, session=sess)

    model.load(MODELPATH)

    print("Model loaded")

    correct_values =[]
    wrong_values=[]
    not_found =[]





    speakers =np.load(SPEAKERSFILE)

    result_list=[]
    for i in os.listdir(PREDICTIONFILESPATH):
        # (rate, sig) = wav.read(PREDICTIONFILESPATH + i)
        # mfccs = mfcc(sig, rate)
        wave, sr = librosa.load(PREDICTIONFILESPATH + i, mono=True)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc = np.pad(mfcc, ((0, 0), (0, 1077 - len(mfcc[0]))), mode='constant', constant_values=0)
        mfcc.flatten()
    # demo=data.load_wav_file(data.test_path + demo_file)

        result = model.predict([mfcc])

    # ress = model.predict_label([mfcc])
    # ress = model.evaluate(X,Y,batch_size=128)

        resultss = data.prob_to_result(result, speakers)

        if len(resultss)<1:
            print('Not found')
            not_found.append(i)
        # temp_result_list=[]
        for j in resultss:
            temp_result =[]
            temp_result.append(i)
            temp_result.append(j["Label"])
            temp_result.append(j["Percentage"])
            print("predicted speaker for %s : label=%s : result = %s : total_prediction:%s " % (i,j["Label"] ,j["Percentage"],len(resultss)))  # ~ 97% correct
            if(i.split('_')[1]==j["Label"]):
                correct_values.append(j["Label"])
            else:
                wrong_values.append(j["Label"])
            result_list.append(temp_result)

    # with open(OUTPUTFILECSV,'rb') as csvWriter:
    #     csvWriter.writelines(result_list)
    np.savetxt(OUTPUTFILECSV, result_list, fmt='%s')
    import csv

    with open(OUTPUTFILECSV, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in result_list:

            filewriter.writerow(i)


    print(len(correct_values))
    print(len(wrong_values))
    print(len(not_found))
    print(not_found)


global args
args = get_args()
INPUTFOLDERPATH =args.inputfolder
PREDICTIONFILESPATH=args.predictionfile
DICTIONARYPATH =args.dictionary
SPEAKERSFILE =args.speakersfile
EPOCHSNO =args.epochs
BATCHSIZE=args.batchsize
LEARNINGRATE=args.learningrate
OPTIMIZER=args.optimizer
ACTIVATION=args.activation
LOOPVALUE=args.loopvalue
MODELPATH =args.modelpath
LOSSFUNCTION = args.lossfunction
OUTPUTFILECSV = args.outputfile
print(args)
predict_speaker()









