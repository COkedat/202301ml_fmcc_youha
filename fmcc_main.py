# os랑 넘파이 불러오기
import os
import numpy as np

# 사운드용
import librosa as lr
import soundfile as sf

# 디노이저 함수 불러오기
from fmcc_denoiser import denoiseWav

# 학습함수, 테스트함수 불러오기
from fmcc_train_predict import train_set, predict_set

# 매개변수 받기용
import sys


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
        print(destinationPath+" done... \r", end="")

        
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
            denoiseWav(fileName, dest)
            print(dest+" done... \r", end="")
    

#평가용 wav로 변환
def readTestWav(train_path = "./fmcc_test900.ctl"):
    #파일명 비어있을 경우
    if(len(train_path)==0):
        print("파일명이 비어있으므로 fmcc_test900.ctl 으로 진행")
        train_path="voice_test.csv"
    sample_rate = 16000 # 16KHz
    data_length = sample_rate * 60 # 16KHz * 60

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
        print(destinationPath+" done... \r", end="")

# 평가용 wav 잡음 삭제 -> R_test 저장
def denoiseTestWav():
    test_wav_path = 'raw16k/test_wav'
    test_wav_denoise_path = 'R_test'
    file_list = os.listdir(test_wav_path)
    wav_files = [file for file in file_list if file.endswith('.wav')]
    
    for wav_file in wav_files:
        fileName = test_wav_path + "/" + wav_file
        dest = test_wav_denoise_path + "/" + wav_file[0:14] + "_denoise.wav"
        denoiseWav(fileName, dest)
        print(dest+" done... \r", end="")


def main():
    try:
        print(sys.argv[1])
    except:
        print("매개변수가 없?는듯???")
        input("아무 키나 입력해서 종료")
        exit()

    if(sys.argv[1]=="trainWav"):
        print("학습용 raw 파일 변환을 시작합니다")
        readTrainWav()
        print("학습용 wav 파일 잡음 제거를 시작합니다")
        denoiseTrainWav()

    elif(sys.argv[1]=="testWav"):
        path=input("테스트용 ctl 파일명을 입력해주세요 : ")
        print("테스트용 raw 파일 변환을 시작합니다")
        readTestWav(path)
        print("테스트용 wav 파일 잡음 제거를 시작합니다")
        denoiseTestWav()

    elif(sys.argv[1]=="train"):
        path=input("학습용 csv 파일명을 입력해주세요 : ")
        print("학습을 시작합니다")
        train_set(path)

    elif(sys.argv[1]=="predict"):
        path=input("테스트용 csv 파일명을 입력해주세요 : ")
        print("평가를 시작합니다")
        predict_set(path)

    else:
        print("? 어캐옴??")


    input("아무 키나 입력해서 종료")
    exit()



if __name__ == "__main__":
	main()


'''

앞으로 구현해야하는것들
1. 전처리
- 지나치게 큰잡음 제거 (ㅇㅋ)
- 볼륨 조정(?)


2. 학습
- 전처리한 사운드들을 기반으로 학습함 (ㅇㅋ)


3. 학습 테스트
- 물론 테스트용 데이터도 잡음이 섞여있기 때문에 전처리 후 평가해야함 (ㅇㅋ)
- 대충 됨

'''
