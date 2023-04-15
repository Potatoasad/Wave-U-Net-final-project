import torch.nn as nn
import torch

class DownsamplingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=15, downsampling=2):
        super(DownsamplingBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size)//2)
        self.pool1 = nn.MaxPool1d(kernel_size=downsampling, stride=2, padding=0)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        shortcut = x
        x = self.pool1(x)
        return x, shortcut
        
class BottleNeckConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=15):
        super(BottleNeckConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size)//2)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
    
class UpsamplingBlock(nn.Module):
    def __init__(self, input_channels, short_channels, output_channels, kernel_size=5, upsampling=2):
        super(UpsamplingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels=(input_channels + short_channels), out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size)//2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU()
        
        
    def make_x_like_y(self, x, y):
        return x[:, :, 0:y.shape[-1]].contiguous()
     
    def forward(self, x, shortcut):
        x = self.upsample1(x)
        combined = torch.cat([x, self.make_x_like_y(shortcut, x)], dim=1)
        x = self.conv1(combined)
        x = self.relu(x)
        return x

class FinalLayer(nn.Module):
    def __init__(self, input_channels, output_channels, short_channels, kernel_size=1, other_audio_index=2, mono=True):
        super(FinalLayer, self).__init__()
        if mono:
            self.remove_channel = 1
        else:
            self.remove_channel = 2
        self.conv1 = nn.ConvTranspose1d(in_channels=(input_channels + short_channels), out_channels=output_channels-self.remove_channel, kernel_size=kernel_size, stride=1, padding=(kernel_size)//2)
        self.tanh = nn.Tanh()
        self.other_audio_index = other_audio_index
        
    def make_x_like_y(self, x, y):
        return x[:, :, 0:y.shape[-1]].contiguous()
        
    def forward(self, x, original_audio):
        combined = torch.cat([x, self.make_x_like_y(original_audio, x)], dim=1)
        x = self.conv1(combined)
        x = self.tanh(x)
        summed_audio = torch.sum(x, dim=1)
        aud_shape = summed_audio.shape
        summed_audio = summed_audio.view((aud_shape[0], 1, *aud_shape[1::]))
        other = original_audio - summed_audio
        return torch.cat([x[:,0:self.other_audio_index, :], other, x[:, self.other_audio_index::, :]], dim=1)


def design_unet(L = 5,              # 5 total layers
                Fc = 16,            # 16 additional Filter channels in each layer
                in_channels = 1,    # Only 1 mono mixture
                out_channels = 4    # Output 4 different sources
               ):
    C_down = lambda i: i*Fc if i != 0 else in_channels
    C_up = lambda i: i*Fc if i != 0 else out_channels

    downs = [( C_down(i) , C_down(i+1) ) for i in range(L-1)]
    bottle = (C_down(L-1), C_down(L))
    final = (C_up(1), in_channels, C_up(0))
    ups = [( C_up(i+1) , C_up(i) , C_up(i)) for i in range(1, L)]
    return downs, bottle, final, ups

def validate_u_net(downs, bottle, final, ups, in_channels, out_channels):
    assert downs[0][0] == in_channels
    
    for i in range(1,len(downs)):
        assert downs[i-1][1] == downs[i][0]
        
    assert downs[-1][1] == bottle[0]
    
    assert bottle[1] == ups[-1][0]
    
    for i in range(len(ups)-1):
        assert (ups[-1-(i+1)][0] == ups[-1-(i)][2])
        assert (ups[-1-(i)][1] == downs[-1-(i)][1])
        
    assert ups[0][2] == final[0]
    
    assert final[1] == in_channels
    assert final[2] == out_channels
    
def create_down(down_tuple):
    DB = DownsamplingBlock(input_channels=down_tuple[0], output_channels=down_tuple[1], kernel_size=15); 
    return DB

def create_up(up_tuple):
    UP = UpsamplingBlock(input_channels=up_tuple[0], short_channels=up_tuple[1], output_channels=up_tuple[2], kernel_size=5);
    return UP
    

class WaveUNet(nn.Module):
    def __init__(self, L=5, Fc=16, in_channels=1, out_channels=4, mono=True):
        super(WaveUNet, self).__init__()
        self.mono = mono
        downs, bottle, final, ups = design_unet(L, Fc, in_channels, out_channels)
        validate_u_net(downs, bottle, final, ups, in_channels, out_channels)
        self.DSBs = nn.ModuleList([create_down(d) for d in downs])
        self.USPs = nn.ModuleList([create_up(u) for u in ups])
        self.FL = nn.ModuleList([FinalLayer(input_channels=final[0], short_channels=final[1], output_channels=final[2], kernel_size=1, mono=mono)])
        self.bottle_neck = nn.ModuleList([BottleNeckConv(input_channels=bottle[0], output_channels=bottle[1])])
        
    def forward(self, X0):
        shorts = []; Xs = [X0];
        for i in range(len(self.DSBs)):
            X, short = self.DSBs[i](Xs[i])
            Xs.append(X); shorts.append(short)
            
        X = self.bottle_neck[0](Xs[-1])
        Xs.append(X)

        for i in range(len(self.USPs)):
            reversed_i = len(self.USPs) - i - 1
            X = self.USPs[reversed_i](Xs[i + len(self.DSBs) + 1], shorts[reversed_i])
            Xs.append(X);   
            
        return self.FL[0](Xs[-1], X0)