from datetime import datetime, timezone
import math
import pytest
from scripts import curve_data as report


def test_elapsed_series_shares_origin_and_breaks_at_scheduler_restart():
    records = [
        {'utc': '2026-10-13T01:00:00+00:00', 'step': 40, 'loss': 0.3},
        {'utc': '2026-10-13T02:00:00+00:00', 'step': 80, 'loss': 0.2},
        {'utc': '2026-10-13T03:00:00+00:00', 'step': 10, 'loss': 0.1},
    ]
    x, y = report.curve_series(records, 'loss', 'elapsed', datetime(2026,10,13,tzinfo=timezone.utc))
    assert x[:2] == [1.0, 2.0] and x[-1] == 3.0
    assert y[:2] == [0.3, 0.2] and y[-1] == 0.1
    assert math.isnan(x[2]) and math.isnan(y[2])
    second_run = [{'utc':'2026-10-13T02:30:00+00:00','step':1,'loss':0.8}]
    assert report.curve_series(second_run,'loss','elapsed',datetime(2026,10,13,tzinfo=timezone.utc))[0] == [2.5]


def test_elapsed_series_refuses_missing_timestamp_instead_of_inventing_time():
    with pytest.raises(ValueError, match='UTC timestamp'):
        report.curve_series([{'step':1,'loss':0.3}], 'loss', 'elapsed', datetime.now(timezone.utc))
