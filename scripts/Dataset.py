import torch
import h5py
from .Audio import *
from .Track import *
from torch.utils.data import Dataset

class SourceSeperationDataset(Dataset):
    def __init__(self, hdf5_dir, transform=None, target_transform=None):
        super().__init__()
        self.hdf5_dir = hdf5_dir
        self.transform = transform
        self.target_transform = target_transform
        
        with h5py.File(self.hdf5_dir, "r") as f:
            all_groups = f.keys()
            self.length = len(all_groups)
            self.rate = f.attrs["samplerate"]
            self.channels = f.attrs["channels"]
            self.instruments = f.attrs["instruments"]
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_dir, 'r') as hf:
            source_audio = hf[f'{idx}/targets'][:]
            mix_audio = hf[f'{idx}/inputs'][:]
            inputs = torch.tensor(mix_audio)
            outputs = torch.tensor(source_audio)
            
            # Apply any transformations that are defined
            if self.transform:
                inputs = self.transform(inputs)
            
            if self.target_transform:
                outputs = self.transform(outputs)
                
            
            input_output = (inputs, outputs)
        return input_output
    
class SourceSeperationDatasetAudio(Dataset):
    def __init__(self, hdf5_dir, transform=None, target_transform=None):
        super().__init__()
        self.hdf5_dir = hdf5_dir
        self.transform = transform
        self.target_transform = target_transform
        
        with h5py.File(self.hdf5_dir, "r") as f:
            all_groups = f.keys()
            self.length = len(all_groups)
            self.rate = f.attrs["samplerate"]
            self.channels = f.attrs["channels"]
            self.instruments = f.attrs["instruments"]
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return AudioStemSection.from_hdf5_file(self.hdf5_dir, idx)