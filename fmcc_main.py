import os
import numpy as np
import librosa as lr
import soundfile as sf
import os
from rpy2.robjects import pandas2ri, packages as robjects
pandas2ri.activate()
stats = packages.importr('stats')

# 학습파일 
def readTrainFiles():
    sample_rate = 16000 # 16KHz
    data_length = sample_rate * 60 # 16KHz * 60

    # 학습할 파일명들 저장된 ctl 파일 읽기
    train_path = "./fmcc_train.ctl"

    # 읽어서 filenames에 저장
    with open(train_path) as f:
        trainNames = f.read().splitlines()

    # wav로 전부 변환
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

#R 스크립트 불러오기
def writeCSV():
    r = robjects.r
    r.source('R/extractfeatures_from_wav.R')

# wav 생성 안했으면 아래거 주석 해제하셈
#readTrainFiles()


writeCSV()


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
