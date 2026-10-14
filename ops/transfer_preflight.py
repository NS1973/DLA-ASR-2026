"""Cheap receiver reserve check before rehashing or reopening an incomplete TAR.

The receiver repeats its authoritative checks per member. This conservative
preflight adds startup margins and cannot authorize a policy change by itself.
"""
GiB=2**30


def transfer_ready(state,host_reserve=32*GiB):
    if host_reserve not in (24*GiB,32*GiB):
        raise ValueError('Unexpected host reserve policy')
    requirements={'available':host_reserve+GiB//2,
                  'container_headroom':9*GiB,
                  'scratch_free':5*GiB+64*2**20,
                  'home_free':20*GiB,'root_free':GiB}
    for name,minimum in requirements.items():
        if state[name]<minimum:
            return False,name
    return True,None
