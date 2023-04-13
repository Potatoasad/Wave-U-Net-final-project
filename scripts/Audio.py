import numpy as np
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from IPython.display import Audio
import matplotlib.pyplot as plt
import librosa


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

class AudioSection:
    """
    Assumes a shape for the input audio as (stereochannel , sample)
    """
    def __init__(self, audio, rate=44100):
        self.audio = audio
        self.audio_shape = audio.shape
        self.rate = rate
        self.resampler = None
       
    @property
    def is_mono(self):
        return (self.audio.shape[1] == 1)
        
    def play(self):
        if isinstance(self.audio, torch.Tensor):
            if self.is_mono: 
                return Audio(data=self.audio.reshape(np.prod(audio.shape)).numpy(), rate=self.rate)
            return Audio(data=self.audio.numpy(), rate=self.rate)
        elif isinstance(self.audio, np.ndarray):
            if self.is_mono:
                aud = self.audio.T
                return Audio(data=aud.reshape(np.prod(aud.shape())), rate=self.rate)
            return Audio(data=self.audio.T, rate=self.rate)
        else:
            raise NotImplementedError(f"Don't know how to play a {type(self.audio)}")
    
    def resample(self, new_rate):
        self.resampler = T.Resample(self.rate, new_rate, dtype=self.audio.dtype)
        self.audio = self.resampler(self.audio)
        self.rate = new_rate
        self.audio_shape = self.audio.shape
    
    @property
    def duration(self):
        return self.audio_shape[1]/self.rate
    
    @property
    def duration_mmss(self):
        total_seconds = self.audio_shape[1]/self.rate
        remaining_seconds = total_seconds % 60
        minutes = total_seconds // 60
        return (minutes, remaining_seconds)
    
    def convert_to_mono(self):
        aud = librosa.to_mono(self.audio.numpy())
        self.audio = torch.tensor(aud.reshape((1,*aud.shape)), dtype=self.audio.dtype, device=self.audio.device)
    
    def cut_into_sections_based_on_duration(self, duration, keep_end=False):
        #nsamps = int(duration*self.rate)
        #total_sections = self.audio_shape[1] // nsamps
        #remainder = (self.audio_shape[1] - int(total_sections)*int(nsamps))
        #audios = [AudioSection(self.audio[:, (i*nsamps):((i+1)*nsamps)], self.rate) for i in range(total_sections)]
        #if keep_end:
        #    audios + [AudioSection(self.audio[:, (total_sections)*nsamps::], self.rate)]
        #return audios
        nsamps = int(duration*self.rate)
        return self.cut_into_sections_based_on_samples(nsamps, keep_end=keep_end)
    
    def cut_into_sections_based_on_samples(self, nsamps, keep_end=False):
        #nsamps = int(duration*self.rate)
        total_sections = self.audio_shape[1] // nsamps
        remainder = (self.audio_shape[1] - int(total_sections)*int(nsamps))
        audios = [AudioSection(self.audio[:, (i*nsamps):((i+1)*nsamps)], self.rate) for i in range(total_sections)]
        if keep_end:
            audios + [AudioSection(self.audio[:, (total_sections*nsamps)::], self.rate)]
        return audios
    
    def spectogram(self, n_fft = 1024, win_length = None, hop_length = 512):
        spectrogram = T.Spectrogram(
                                    n_fft=n_fft,
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    center=True,
                                    pad_mode="reflect",
                                    power=2.0,
                                )
        return spectrogram(self.audio)
    
    def plot_spectogram(self):
        spec = self.spectogram()
        for i in range(spec.shape[0]):
            plot_spectrogram(spec[i])
            
    def __str__(self):
        dur = self.duration_mmss
        return f"Audio file of duration {dur[0]} minutes {np.round(dur[1],2)} seconds at a sample rate of {self.rate} Hz"
    
    
class AudioStemSection:
    def __init__(self, stems):
        self.stems = stems
        self.rate = stems[0].rate
        self.is_mono = stems[0].is_mono
        self.instrument_to_index = {"mixture" : 0,  
                                    "drums" : 1, 
                                    "bass" : 2,
                                    "other" : 3,
                                    "vocals" : 4}
        
    def convert_to_mono(self):
        self.is_mono = True
        for i in range(len(self.stems)):
            self.stems[i].convert_to_mono()
            
    def resample(self, rate=None, ds=None):
        if rate is None:
            rate = self.rate // ds
        self.rate = rate
        for i in range(len(self.stems)):
            self.stems[i].resample(rate)
            
    def cut_into_sections_based_on_samples(self, nsamps, **kwargs):
        N_stems = len(self.stems)
        cuts = [self.stems[i].cut_into_sections_based_on_samples(nsamps, **kwargs) for i in range(N_stems)]
        return [AudioStemSection([cuts[i_stem][j_samp] for i_stem in range(N_stems)]) for j_samp in range(len(cuts[0]))]
    
    def cut_into_sections_based_on_duration(self,duration, **kwargs):
        N_stems = len(self.stems)
        cuts = [self.stems[i].cut_into_sections_based_on_duration(duration, **kwargs) for i in range(N_stems)]
        return [AudioStemSection([cuts[i_stem][j_samp] for i_stem in range(N_stems)]) for j_samp in range(len(cuts[0]))]
    
    @property
    def audio(self):
        return torch.stack([self.stems[i].audio for i in range(len(self.stems))])
    
    def __getitem__(self, i):
        if isinstance(i, str):
            i = self.instrument_to_index[i.lower()]
            return self.stems[i]
        if isinstance(i, int):
            return self.stems[i]
        if isinstance(i, list):
            return [self.__getitem__(x) for x in i]
    
    def __len__(self):
        return len(self.stems)

    def play(self, i):
        return self[i].play()
    
    def get_stacked_audio(self, instruments=[0,1,2,3,4]):
        return torch.stack([self[i].audio for i in instruments])
    
    def get_train_test(self):
        return (self[0], self[[1,2,3,4]])
    
    @classmethod
    def from_hdf5_file(cls, hdf_dir, i):
        with h5py.File(hdf_dir, 'r') as hf:
            source_audio = hf[f'{i}/targets'][:]
            mix_audio = hf[f'{i}/inputs'][:]
            all_mix = torch.concat([torch.tensor(mix_audio), torch.tensor(source_audio)], dim=0)
            all_stems = [AudioSection(all_mix[i,:,:], rate=hf.attrs["samplerate"]) for i in range(len(hf.attrs["instruments"]) + 1)]
        return cls(all_stems)