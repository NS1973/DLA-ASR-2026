"""Separate offline run showing all four augmentations on real training audio."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.asr.data import CustomDirDataset,load_audio,augment_waveform
from src.asr.runtime import RunLog,log_media,seed_all

torch.set_num_threads(2)
seed_all(42)
parser=argparse.ArgumentParser()
parser.add_argument('--profile',choices=['legacy','effective_v2'],default='legacy')
args=parser.parse_args()
directory = Path('/work/runs/augmentation-proof' + ('-v2' if args.profile=='effective_v2' else ''))
logger = RunLog(directory,{'purpose':'separate augmentation demonstration','seed':42,'augmentation_profile':args.profile})
entries = CustomDirDataset('/work/data/bootstrap').entries[:3]
log_media(entries,directory,logger,augmentation_profile=args.profile)
for entry in entries:
    original = load_audio(entry['path'])
    for kind in ['gain','noise','speed','reverb']:
        wave = augment_waveform(original,kind,profile=args.profile)
        assert torch.isfinite(wave).all()
        assert wave.shape != original.shape or not torch.equal(wave,original)
        logger.emit({'event':'augmentation_check','uid':entry['uid'],'kind':kind,
                     'original_samples':len(original),'transformed_samples':len(wave),
                     'rms':float(wave.square().mean().sqrt()),'peak':float(wave.abs().max())})
logger.emit({'event':'augmentation_proof_complete','records':len(entries),'augmentations':4})
if logger.wandb:
    logger.wandb.finish()
