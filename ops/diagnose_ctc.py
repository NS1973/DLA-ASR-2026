"""Read-only diagnosis of a saved failing batch, without using a GPU."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.asr.data import SpeechDataset, LengthBatchSampler, librispeech_entries
from src.asr.model import DeepSpeech2CTC

torch.set_num_threads(2)
entries = librispeech_entries('/work/data/datasets','train-clean-100')
invalid_original = []
for entry in entries:
    text = entry['text']
    required = len(text)+sum(a==b for a,b in zip(text,text[1:]))
    frames = round(entry['duration']*16000)//160+1
    if (frames+3)//4 < required:
        invalid_original.append(entry['uid'])
print(json.dumps({'original_records':len(entries),'infeasible_original_records':invalid_original}))
dataset = SpeechDataset(entries,augment=True,seed=1)
dataset.epoch[0] = 3
sampler = LengthBatchSampler(entries,32,1,True)
sampler.epoch = 3
# Step3541 completed => next batch index3541-3*892=865.
indices = list(sampler)[3541-3*len(sampler)]
for index in indices:
    item = dataset[index]
    target = item['encoded']
    required = len(target)+int((target[1:]==target[:-1]).sum())
    frames = item['spectrogram'].shape[-1]
    available = int(DeepSpeech2CTC.transform_input_lengths(torch.tensor([frames]))[0])
    if available < required:
        original = SpeechDataset([entries[index]])[0]
        print(json.dumps({'uid':item['uid'],'index':index,'original_seconds':entries[index]['duration'],
                          'augmented_seconds':len(item['waveform'])/16000,'frames':frames,
                          'available':available,'required':required,
                          'original_frames':original['spectrogram'].shape[-1]},indent=2))
