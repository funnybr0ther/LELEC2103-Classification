import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import sounddevice as sd
import torch

from torch.utils.data import DataLoader, Dataset, random_split
from os import walk
from torchaudio import transforms

nothing_path = "./room_sounds/"
data_path = "./ESC-50/audio/"
meta_path = "./ESC-50/meta/"

class AudioUtil:
    @staticmethod
    def open(path):
        return torchaudio.load(path)
          
    @staticmethod
    def toFile(aud,path):
        sig, sr = aud
        torchaudio.save(path,sig,sr)

    @staticmethod
    def displayTime(aud):
        sig, sr = aud
        num_channels,num_samples = sig.shape

        if(num_channels != 1):
            raise Exception("Can't display multi-channel sound.")

        t = np.linspace(0,num_samples/sr,num_samples)
        plt.plot(t,sig.numpy().ravel())

        plt.show()

    @staticmethod
    def playSound(aud):
        sig, sr = aud
        sd.play(0.1*sig.numpy().ravel(),sr,blocking=True)

    @staticmethod
    def toMelSpec(aud,n_fft = 512, n_mels = 20, truncate_to=0):
        sig,sr=aud
        num_channels,num_samples = sig.shape
        if(truncate_to>0):
            sig = sig[:,:truncate_to*n_fft]
        
        melspec = transforms.MelSpectrogram(sample_rate = sr,n_fft=n_fft,hop_length=n_fft,n_mels=n_mels,center=False)(sig)
        # melspec = transforms.AmplitudeToDB()(melspec)
        melspec = torch.abs(melspec)
        if melspec.max() != 0:
            melspec = melspec / melspec.max()
        return melspec
    
    @staticmethod
    def displayMelspec(melspec, aud=None):
        if aud != None:
            sig,sr = aud
            num_channels,num_samples = sig.shape

            tmax = num_samples/sr

        else:
            tmax = 0.5
        t = np.linspace(0,tmax,5)
        
        plt.imshow(melspec.numpy()[0], interpolation='nearest', origin="lower",aspect="auto")
        plt.colorbar()
        xmin,xmax = plt.xlim()
        plt.xticks(t*xmax/tmax,["{:10.4f}".format(i) for i in t])
        plt.show()

    @staticmethod
    def rechannel(aud):
        sig, sr = aud
        
        num_channels, num_samples  = sig.shape
        if(num_channels == 1):
            return aud
        
        else:
            ## Average the channels
            sig = sig.mean(axis = 0)
            sig = sig.reshape((1,sig.shape[0]))
            return (sig, sr)

    @staticmethod
    def resample(aud, new_sr):
        sig, sr = aud

        num_channels, num_samples  = sig.shape
        if(num_channels != 1):
            raise Exception("Can't apply resample to multi-channel sound.")
        
        resig = torchaudio.transforms.Resample(sr,new_sr)(sig)
    
        return ((resig,new_sr))
        
    @staticmethod
    def sliceAudio(aud, trunc, index, trunc_type = "ms",bootstrap = False,bootstrapSeed=0):
        sig, sr = aud
        num_channels, num_samples  = sig.shape
        
    
        if(num_channels != 1):
            raise Exception("Can't apply slice to multi-channel sound.")

        if not bootstrap:
            if trunc_type == "ms":
                samples_per_slice = int(trunc/1000 * sr)
                if samples_per_slice*index > num_samples:
                    raise Exception("Can't extract a slice with index {} from this audio file.".format(index)) 
                
            elif trunc_type == "samples":
                samples_per_slice = trunc
                if samples_per_slice*index > num_samples:
                    raise Exception("Can't extract a slice with index {} from this audio file.".format(index))
            else:
                raise Exception('''Unknown truncation type, use "ms" or "samples"''')

            return (sig[:,index*samples_per_slice : (index+1)*samples_per_slice],sr)
        else:
            if trunc_type == "ms":
                samples_per_slice = int(trunc/1000 * sr)
                if samples_per_slice > num_samples:
                    raise Exception("Can't extract slices of such length from this audio file.") 
            elif trunc_type == "samples":
                samples_per_slice = trunc
                if samples_per_slice > num_samples:
                    raise Exception("Can't extract slices of such length from this audio file.")
            else:
                raise Exception('''Unknown truncation type, use "ms" or "samples"''')
        
            randomPermut = np.random.RandomState(bootstrapSeed).randint(0,num_samples-samples_per_slice-1,index+1)[index]
            return ((sig[:,randomPermut : randomPermut+samples_per_slice],sr))
        
    @staticmethod
    def augment_timeShift(aud,shift_limit,seed):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(np.random.RandomState(seed).random(size=1) * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def augment_spectralMask(spec,max_mask_pct,n_freq_masks=1,n_time_masks=1):
        """
        TODO: Don't use for now, must use a seed to work properly
        """
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels

        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec,mask_value)
        
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec,mask_value)
        
        return aug_spec

    @staticmethod
    def augment_timeNoise(aud,noise_power):
        """
        TODO: Use a seed????
        """
        sig,sr = aud
        num_channels, num_samples  = sig.shape

        sig_aug = sig + np.random.normal(0,scale=noise_power**0.5,size=num_samples).astype('f')

        return (sig_aug,sr)
        
    @staticmethod
    def augment_timeLowPass(aud,cutoff_freq):
        """
        Highly unstable at high cutoff frequencies!!!! (>sr/2) TODO: Fix this
        """
        sig,sr = aud
        sig_aug = torchaudio.functional.lowpass_biquad(sig,sr,cutoff_freq=cutoff_freq)

        return (sig_aug,sr)


    @staticmethod
    def normalize_specgram(spec):
        return spec/(np.mean(np.mean(np.abs(spec[0].numpy())**2,axis=0),axis=0) ** 0.5)

    @staticmethod
    def convert_room_sounds():
        filenames = next(walk("./source_room_sounds/"), (None, None, []))[2]  # [] if no file
        i = 0
        for filename in filenames:
            opened_file = AudioUtil.open("./source_room_sounds/" + filename)
            opened_file = AudioUtil.rechannel(opened_file)
            number_of_chunks = int(opened_file[0].shape[1] / (5*opened_file[1]))

            for j in range(number_of_chunks):
                sub_file = AudioUtil.sliceAudio(opened_file,5000,j)
                AudioUtil.toFile(sub_file,nothing_path + str(i) + ".wav")
                i+=1
        return i

    
class SoundDS(Dataset):
    def __init__(self,df, data_path, bootstrap, bootstrapSeed = np.random.randint(0,20)):
        self.df = df
        self.data_path = str(data_path)
        self.samplesPerFile= 10
        self.windowSize = 512
        self.duration = self.windowSize*self.samplesPerFile
        self.sr = 11025
        self.channel = 1
        self.bootstrap = bootstrap
        self.augmentations = ["timeShift", "noise", "lowPass", "mask"]
        self.bootstrapSeed = bootstrapSeed
        self.nothing_size = AudioUtil.convert_room_sounds()
    
    def __len__(self):
        return len(self.df)*self.samplesPerFile + self.nothing_size*self.samplesPerFile
        
    def __getitem__(self, idx):
        index = idx // self.samplesPerFile
        loaded_file = self.getAudio(idx)

        melSpecGram = AudioUtil.toMelSpec(loaded_file,self.windowSize,20,0)

        # melSpecGram = AudioUtil.normalize_specgram(melSpecGram)
        # if "mask" in self.augmentations:
            # melSpecGram = AudioUtil.augment_spectralMask(melSpecGram,0.1,1,1)
        if(idx >= len(self.df)*self.samplesPerFile):
            class_id = 5
        else:
            class_id = self.df["category"].iloc[index]

        return melSpecGram,class_id

    def getAudio(self,idx):
        if(idx < len(self.df)*self.samplesPerFile):
            index = idx // self.samplesPerFile
            subindex = idx % self.samplesPerFile

            audio_file = self.df["filename"].iloc[index]
            loaded_file = AudioUtil.open(self.data_path + audio_file)
        else:
            idx -= len(self.df)*self.samplesPerFile

            index = idx // self.samplesPerFile
            subindex = idx % self.samplesPerFile

            audio_file = str(index) + ".wav"
            loaded_file = AudioUtil.open(nothing_path + audio_file)
            
        loaded_file = AudioUtil.rechannel(loaded_file)
        loaded_file = AudioUtil.resample(loaded_file,11025)
        loaded_file = AudioUtil.sliceAudio(loaded_file,self.duration,subindex,"samples",self.bootstrap,self.bootstrapSeed)
        if "timeShift" in self.augmentations:
            loaded_file = AudioUtil.augment_timeShift(loaded_file,1.0,idx*self.bootstrapSeed)
        if "noise" in self.augmentations:
            loaded_file = AudioUtil.augment_timeNoise(loaded_file,0.0000)
        # if "lowPass" in self.augmentations:
        #     loaded_file = AudioUtil.augment_timeLowPass(loaded_file,5000)
        
        return loaded_file

