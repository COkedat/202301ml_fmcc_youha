import torch
import torchaudio
import soundfile as sf
from denoiser import pretrained
from denoiser.dsp import convert_audio

#torch cpu로 변경 (cuda는 torch 버젼 문제나 그런거 생길 수 있어서 그냥 CPU로 통일)
def denoiseWav(fileName, destName):
    model = pretrained.dns64()
    wav, sr = torchaudio.load(fileName)
    with torch.no_grad():
        denoised = model(wav[None])[0].cpu()
    torchaudio.save(destName, denoised, 16000, bits_per_sample=16)
