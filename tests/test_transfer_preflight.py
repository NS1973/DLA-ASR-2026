import pytest
from ops.transfer_preflight import transfer_ready

GiB=2**30


def state(**overrides):
    result=dict(available=34*GiB,container_headroom=12*GiB,
                scratch_free=10*GiB,home_free=21*GiB,root_free=2*GiB)
    result.update(overrides)
    return result


def test_default_reserve_waits_before_expensive_hashing_and_explicit_lower_policy_is_separate():
    assert transfer_ready(state())[0]
    assert not transfer_ready(state(available=31*GiB))[0]
    assert transfer_ready(state(available=31*GiB),host_reserve=24*GiB)[0]
    with pytest.raises(ValueError):transfer_ready(state(),host_reserve=0)


@pytest.mark.parametrize('field,value',[('container_headroom',8*GiB),('scratch_free',5*GiB),('home_free',19*GiB),('root_free',GiB-1)])
def test_lower_host_policy_never_bypasses_other_reserves(field,value):
    ready,reason=transfer_ready(state(**{field:value}),host_reserve=24*GiB)
    assert not ready and reason
