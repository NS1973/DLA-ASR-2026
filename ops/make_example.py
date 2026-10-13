"""Bundle a small real training utterance for the inference demonstration."""
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
dataset = root/'data/downloads/bootstrap'
audio = sorted((dataset/'audio').glob('*.flac'))[0]
text = dataset/'transcriptions'/f'{audio.stem}.txt'
destination = root/'delivery/example.zip'
destination.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(audio,'audio/'+audio.name)
    archive.write(text,'transcriptions/'+text.name)
    archive.writestr('source.json',json.dumps({'dataset':'LibriSpeech train-clean-100','uid':audio.stem,
                                               'purpose':'inference demo, not held-out evaluation'}))
print(destination)
