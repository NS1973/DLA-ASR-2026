"""Validate closed training receipts without assuming a previous campaign schema."""
from datetime import datetime, timezone
import math


def validate_terminal_proof(proof, expected_stage_count):
    states = proof.get('states', [])
    if len(states) != expected_stage_count or not states:
        raise ValueError('Terminal receipt has an incomplete stage inventory')
    if 'all_owned_stage_pids_terminal' in proof:
        if proof['all_owned_stage_pids_terminal'] is not True or proof.get('pending') != []:
            raise ValueError('Terminal receipt still has pending stages')
        runs = [item['run'] for item in states]
        active = proof.get('active', [])
        if len(set(runs)) != len(runs) or not active or not set(active).issubset(runs):
            raise ValueError('Terminal receipt has duplicate or missing active stages')
        completed = float(proof['completed_unix'])
        cutoff = float(proof['cutoff_unix'])
        if not math.isfinite(completed) or not math.isfinite(cutoff) or completed < cutoff:
            raise ValueError('Terminal receipt precedes the training cutoff')
        verified = datetime.fromtimestamp(completed, timezone.utc)
        for item in states:
            status = item['status']
            if item['live'] is not False or status['event'] != 'train_exit':
                raise ValueError('An owned trainer is not terminal')
            stamp = datetime.fromisoformat(status['utc'])
            if stamp.tzinfo is None or stamp > verified:
                raise ValueError('Stage exit is later than terminal verification')
        return verified
    if proof.get('all_four_owned_stage_pids_terminal') is not True or expected_stage_count != 4:
        raise ValueError('Unknown or contradictory legacy terminal receipt')
    if any(item.get('live_owned_trainer') is not False or item.get('event') != 'train_exit'
           for item in states):
        raise ValueError('A legacy owned trainer is not terminal')
    verified = datetime.fromisoformat(proof['verified_utc'])
    if verified.tzinfo is None:
        raise ValueError('Terminal verification needs an explicit timezone')
    return verified
