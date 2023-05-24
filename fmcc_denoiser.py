import torch
import torchaudio
import soundfile as sf
from denoiser import pretrained
from denoiser.dsp import convert_audio

#torch cuda로 설치되어있어야함
#걍 cuda 삭제함
def denoiseWav(fileName, destName):
    model = pretrained.dns64()#.cuda()
    wav, sr = torchaudio.load(fileName)
    #wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0].cpu()
    torchaudio.save(destName, denoised, 16000, bits_per_sample=16)


# 훈련 데이터 잡음 삭제 -> R_train 저장
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

# 테스트 데이터 잡음 삭제 -> R_test 저장
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
        denoiseWav(fileName, dest)
