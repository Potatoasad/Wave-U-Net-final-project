import torch
from .Audio import *

class Track:
    def __init__(self, track):
        self.track = track
        self.stems = track.stems
        
    def get_audio_portion(self, stem_channel, stems, rate, ds=1):
        audio_stem = torch.tensor(stems[stem_channel,:,:].T, dtype=torch.float32)
        audio = AudioSection(audio_stem, rate)
        audio.resample(audio.rate//ds)
        return audio
        
    def get_stem_section(self, ds=1):
        audio = self.stems
        rate = self.track.rate
        self.tracks = [self.get_audio_portion(j, audio, rate, ds=ds) for j in range(5)]
        return AudioStemSection(self.tracks)