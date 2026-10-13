"""Measured plot coordinates, independent of optional rendering dependencies."""
from datetime import datetime


def timestamp(record):
    try:
        value = datetime.fromisoformat(record['utc'])
        if value.tzinfo is None:
            raise ValueError('Timezone missing')
        return value
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('A timezone-aware UTC timestamp is required for elapsed curves') from error


def curve_series(records, key, time_axis='step', origin=None):
    if time_axis not in {'step', 'elapsed'}:
        raise ValueError('Unknown curve time axis')
    x, y, previous_step = [], [], None
    for record in records:
        step = record['step']
        if previous_step is not None and step <= previous_step:
            x.append(float('nan'))
            y.append(float('nan'))
        coordinate = step if time_axis == 'step' else (timestamp(record) - origin).total_seconds() / 3600
        x.append(coordinate)
        y.append(record[key])
        previous_step = step
    return x, y
