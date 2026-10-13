from concurrent.futures import ThreadPoolExecutor
import pytest
from ops.transfer_parallel import mirror_round


def test_parallel_ranges_are_contiguous_and_bounded():
    ranges=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        end=mirror_round(pool,lambda start,count:ranges.append((start,count)),
                         committed=12,available=31,workers=2,block_size=8)
    assert sorted(ranges)==[(12,8),(20,8)]
    assert end==28
    with ThreadPoolExecutor(max_workers=2) as pool:
        end=mirror_round(pool,lambda start,count:ranges.append((start,count)),
                         committed=end,available=31,workers=2,block_size=8)
    assert end==31
    assert ranges[-1]==(28,3)


def test_failed_block_does_not_advance_confirmed_prefix():
    completed=[]
    def transfer(start,count):
        if start==8:raise ConnectionError('interrupted channel')
        completed.append(start)
    committed=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(ConnectionError):
            committed=mirror_round(pool,transfer,committed,16,2,8)
    assert committed==0
    assert completed==[0]


class LocalSFTP:
    def open(self,path,mode='r'):
        return open(path,mode)
    def stat(self,path):
        from pathlib import Path
        return Path(path).stat()


def test_first_transfer_creates_empty_remote_partial(tmp_path):
    from ops.transfer_parallel import prepare_remote
    dest=tmp_path/'archive.partial'
    offset,complete=prepare_remote(LocalSFTP(),str(dest),str(dest)+'.json')
    assert offset==0 and complete is None
    assert dest.read_bytes()==b''


def test_resume_discards_unacknowledged_holes_and_recognizes_final(tmp_path):
    from ops.transfer_parallel import prepare_remote
    import json
    dest=tmp_path/'archive.partial'
    marker=tmp_path/'ack.json'
    dest.write_bytes(b'good'+b'bad unacknowledged tail')
    marker.write_text(json.dumps({'committed':4}))
    offset,complete=prepare_remote(LocalSFTP(),str(dest),str(marker))
    assert offset==4 and complete is None
    assert dest.read_bytes()==b'good'
    final=tmp_path/'archive'
    dest.rename(final)
    offset,complete=prepare_remote(LocalSFTP(),str(dest),str(marker))
    assert offset==4 and complete==str(final)
