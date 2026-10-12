from src.datasets.custom_dir_dataset import CustomDirDataset

# Original template examples are loaded only when explicitly requested.
def __getattr__(name):
    if name == "CommonVoiceDataset":
        from src.datasets.common_voice import CommonVoiceDataset
        return CommonVoiceDataset
    if name == "CustomDirAudioDataset":
        from src.datasets.custom_dir_audio_dataset import CustomDirAudioDataset
        return CustomDirAudioDataset
    if name == "LibrispeechDataset":
        from src.datasets.librispeech_dataset import LibrispeechDataset
        return LibrispeechDataset
    raise AttributeError(name)
