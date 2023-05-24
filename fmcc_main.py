import os
import numpy as np
import librosa as lr
import soundfile as sf
import fmcc_denoiser
#from rpy2.robjects import pandas2ri, packages as robjects
#pandas2ri.activate()
#stats = packages.importr('stats')

# 학습용 wav로 변환
def readTrainWav():
    sample_rate = 16000 # 16KHz
    data_length = sample_rate * 60 # 16KHz * 60

    # 학습할 파일명들 저장된 ctl 파일 읽기
    train_path = "./fmcc_train.ctl"

    # 읽어서 filenames에 저장
    with open(train_path) as f:
        trainNames = f.read().splitlines()

    # wav로 전부 변환해서 train_wav에 저장
    for target in trainNames:
        if(target[0]=="F"):
            dest="raw16k/train_wav/female/"
        else:
            dest="raw16k/train_wav/male/"
        destinationPath=dest+target[6:19]+".wav"
        target="raw16k/train/"+target+".raw"
        with open(target, 'rb') as tf:
            buf = tf.read()
            buf = buf+b'0' if len(buf)%2 else buf
        pcm_data = np.frombuffer(buf, dtype='int16')
        wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2)
        sf.write(destinationPath, wav_data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
        print(destinationPath+" done...")
        
        '''
    for target in trainNames:
        if(target[0]=="F"):
            dest="R/female/"
        else:
            dest="R/male/"
        destinationPath=dest+target[6:19]+".wav"
        target="raw16k/train/"+target+".raw"
        with open(target, 'rb') as tf:
            buf = tf.read()
            buf = buf+b'0' if len(buf)%2 else buf
        pcm_data = np.frombuffer(buf, dtype='int16')
        wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2)
        sf.write(destinationPath, wav_data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
        print(destinationPath+" done...")
        '''
        
# 학습용 wav 잡음 삭제 -> R_train 저장
def denoiseTrainWav():
        
    train_path = 'raw16k/train_wav'
    train_file_list = os.listdir(train_path)

    for sex in train_file_list:
        if sex == 'female':
            train_wav_path = 'raw16k/train_wav/female'
            train_wav_denoise_path = 'R_train/female'
            file_list = os.listdir(train_wav_path)
            wav_files = [file for file in file_list if file.endswith('.wav')]
    
        elif sex == 'male':
            train_wav_path = 'raw16k/train_wav/male'
            train_wav_denoise_path = 'R_train/male'
            file_list = os.listdir(train_wav_path)
            wav_files = [file for file in file_list if file.endswith('.wav')]
    
        for wav_file in wav_files:
            fileName = train_wav_path + "/" + wav_file
            dest = train_wav_denoise_path + "/" + wav_file[0:13] + "_denoise.wav"
            fmcc_denoiser.denoiseWav(fileName, dest)
    

#평가용 wav로 변환
def readTestWav():
    sample_rate = 16000 # 16KHz
    data_length = sample_rate * 60 # 16KHz * 60

    # 학습할 파일명들 저장된 ctl 파일 읽기
    train_path = "./fmcc_test.ctl"

    # 읽어서 filenames에 저장
    with open(train_path) as f:
        trainNames = f.read().splitlines()

    # wav로 전부 변환해서 train_wav에 저장
    for target in trainNames:
        destinationPath="raw16k/test_wav/"+target+".wav"
        target="raw16k/test/"+target+".raw"
        with open(target, 'rb') as tf:
            buf = tf.read()
            buf = buf+b'0' if len(buf)%2 else buf
        pcm_data = np.frombuffer(buf, dtype='int16')
        wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2)
        sf.write(destinationPath, wav_data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
        print(destinationPath+" done...")

# 평가용 wav 잡음 삭제 -> R_test 저장
def denoisTestWav():
        
    test_path = 'raw16k/test_wav'
    test_file_list = os.listdir(test_path)

    for sex in test_file_list:
        if sex == 'female':
            test_wav_path = 'raw16k/test_wav/female'
            test_wav_denoise_path = 'R_test/female'
            file_list = os.listdir(test_wav_path)
            wav_files = [file for file in file_list if file.endswith('.wav')]
    
        elif sex == 'male':
            test_wav_path = 'raw16k/test_wav/male'
            test_wav_denoise_path = 'R_test/male'
            file_list = os.listdir(test_wav_path)
            wav_files = [file for file in file_list if file.endswith('.wav')]
    
        for wav_file in wav_files:
            fileName = test_wav_path + "/" + wav_file
            dest = test_wav_denoise_path + "/" + wav_file[0:13] + "_denoise.wav"
            fmcc_denoiser.denoiseWav(fileName, dest)
            
#R 스크립트 불러오기
'''
def writeCSV():
    r = robjects.r
    r.source('R/extractfeatures_from_wav.R')
'''


#writeCSV()


'''

앞으로 구현해야하는것들
1. 전처리
- 지나치게 큰잡음 제거
- 볼륨 조정(?)


2. 학습
- 전처리한 사운드들을 기반으로 학습함


3. 학습 테스트
- 물론 테스트용 데이터도 잡음이 섞여있기 때문에 전처리 후 평가해야함
- 대충 그럼

'''
