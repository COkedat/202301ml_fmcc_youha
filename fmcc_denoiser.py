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


