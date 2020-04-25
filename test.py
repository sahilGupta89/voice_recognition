# import tflearn
# # import librosa
# print(tflearn)
# # print(librosa)
#
#
# # from python_speech_features import mfcc
# # from python_speech_features import logfbank
# import scipy.io.wavfile as wav
# from mel_coefficients import mfcc
# from scipy.io.wavfile import read
# (fs,s)=read('D:/VA_Neural_Network/VoiceService/WindowsServerService/bin/Debug/TargetDatabase/birdsandnaturefeb1902-01-various-64kb_Nemo_100.wav.wav')
# s=s[s>0]
# mel_coeff=mfcc(s,fs,12)
# print(mel_coeff)
# print(mel_coeff.shape)
# print(mfcc)
# print(wav)
# # except Exception as ex:
# #  print(ex)
# s = input("Done")

import numpy as np
speakers =np.load('D:/Neural_Network_Research/Resources/Output_Data/speaker_list.npy')
print(speakers)