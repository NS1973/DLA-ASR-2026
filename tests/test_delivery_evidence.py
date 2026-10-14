from copy import deepcopy
from datetime import datetime, timezone

import pytest

from ops.delivery_evidence import validate_terminal_proof


def current_proof():
    return {
        'all_owned_stage_pids_terminal': True,
        'pending': [],
        'active': ['run0', 'run1'],
        'cutoff_unix': 100,
        'completed_unix': 110,
        'states': [
            {'run': f'run{i}', 'live': False,
             'status': {'event': 'train_exit', 'utc': '1970-01-01T00:01:45+00:00'}}
            for i in range(22)
        ],
    }


def test_current_campaign_checks_all_22_terminal_stages():
    assert validate_terminal_proof(current_proof(), 22) == datetime.fromtimestamp(110, timezone.utc)


@pytest.mark.parametrize('change', ['live', 'pending', 'missing', 'duplicate', 'active', 'early', 'late_exit'])
def test_terminal_receipt_rejects_incomplete_or_contradictory_evidence(change):
    proof = deepcopy(current_proof())
    if change == 'live':
        proof['states'][0]['live'] = True
    elif change == 'pending':
        proof['pending'] = ['run0']
    elif change == 'missing':
        proof['states'].pop()
    elif change == 'duplicate':
        proof['states'][1]['run'] = 'run0'
    elif change == 'active':
        proof['active'].append('absent')
    elif change == 'early':
        proof['completed_unix'] = 90
    else:
        proof['states'][0]['status']['utc'] = '1970-01-01T00:03:00+00:00'
    with pytest.raises(ValueError):
        validate_terminal_proof(proof, 22)


def test_legacy_receipt_keeps_its_own_four_stage_contract():
    proof = {'all_four_owned_stage_pids_terminal': True,
             'verified_utc': '2026-10-13T17:41:00+00:00',
             'states': [{'live_owned_trainer': False, 'event': 'train_exit'} for _ in range(4)]}
    assert validate_terminal_proof(proof, 4).isoformat() == proof['verified_utc']
