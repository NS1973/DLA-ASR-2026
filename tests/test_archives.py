import io
import tarfile
import zipfile
import pytest


def test_dataset_archive_rejects_path_traversal(tmp_path):
    from scripts.prepare_custom_dataset import extract
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive,'w') as source:
        source.writestr('../escape.txt','unsafe')
    with pytest.raises(ValueError,match='Unsafe'):
        extract(archive,tmp_path/'out')
    assert not (tmp_path/'escape.txt').exists()


def test_nested_dataset_folder_is_found(tmp_path):
    from scripts.prepare_custom_dataset import extract
    archive = tmp_path/'valid.zip'
    with zipfile.ZipFile(archive,'w') as source:
        source.writestr('dataset/audio/a.wav',b'placeholder')
    assert extract(archive,tmp_path/'out') == tmp_path/'out/dataset'
