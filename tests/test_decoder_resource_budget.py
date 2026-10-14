import pytest
from ops.queue_guards import decoder_worker_budget

GIB = 2**30


def test_decoder_workers_keep_host_reserve_and_workspace():
    assert decoder_worker_budget(40*GIB, 8) == 8
    assert decoder_worker_budget(33*GIB, 8) == 6
    assert decoder_worker_budget(25*GIB, 8) == 0


def test_decoder_budget_never_expands_requested_parallelism():
    assert decoder_worker_budget(50*GIB, 2) == 2
    assert decoder_worker_budget(50*GIB, 1) == 1


def test_decoder_budget_rejects_invalid_measurements():
    with pytest.raises(ValueError):
        decoder_worker_budget(-1, 8)
    with pytest.raises(ValueError):
        decoder_worker_budget(40*GIB, 0)
