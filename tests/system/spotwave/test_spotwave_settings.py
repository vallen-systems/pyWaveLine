from datetime import datetime

from pytest import mark


def test_get_info(sw):
    info = sw.get_info()
    assert info  # valid, non-empty response
    assert info.firmware_version


def test_get_setup(sw):
    assert sw.get_setup()  # valid, non-empty response


def test_get_status(sw):
    assert sw.get_status()  # valid, non-empty response


@mark.parametrize(
    "set_, expect",
    (
        (False, False),
        (True, True),
    ),
)
def test_set_continuous_mode(sw, set_, expect):
    sw.set_continuous_mode(set_)
    assert sw.get_setup().cont_enabled == expect


@mark.parametrize(
    "set_, expect",
    (
        (-1, 0),
        (1000, 1000),
        (100_000, 100_000),
        (1_000_000, 1_000_000),  # TODO: should be set to max of 100_000
    ),
)
def test_set_ddt(sw, set_, expect):
    sw.set_ddt(set_)
    assert sw.get_setup().ddt_seconds == expect / 1e6


@mark.parametrize(
    "set_, expect",
    (
        (-1, 0),
        (0.01, 0.01),
        (2, 2),
        (3600, 3600),
    ),
)
def test_set_status_interval(sw, set_, expect):
    sw.set_status_interval(set_)
    assert sw.get_setup().status_interval_seconds == expect


@mark.parametrize(
    "set_, expect",
    (
        (False, False),
        (True, True),
    ),
)
def test_set_tr_enabled(sw, set_, expect):
    sw.set_tr_enabled(set_)
    assert sw.get_setup().tr_enabled == expect


@mark.parametrize(
    "set_, expect",
    (
        (-1, 1),
        (0, 1),
        (1, 1),
        (10, 10),
        (11.1, 11),
        (1_000_000, 1_000_000),
    ),
)
def test_set_tr_decimation(sw, set_, expect):
    sw.set_tr_decimation(set_)
    assert sw.get_setup().tr_decimation == expect


@mark.parametrize(
    "set_, expect",
    (
        (-5, 0),
        (0, 0),
        (2000, 2000),
        (10000, 10000),  # TODO: limit to 2000 / decimation
    ),
)
def test_set_tr_pretrigger(sw, set_, expect):
    sw.set_tr_pretrigger(set_)
    assert sw.get_setup().tr_pretrigger_samples == expect


@mark.parametrize(
    "set_, expect, ddt",
    (
        (0, 0, 1000),
        (2000, 2000, 1000),  # TODO: limit to DDT
    ),
)
def test_set_tr_postduration(sw, set_, expect, ddt):
    sw.set_ddt(ddt)
    sw.set_tr_postduration(set_)
    assert sw.get_setup().tr_postduration_samples == expect


@mark.parametrize(
    "set_, expect",
    (
        (0, 0),
        (-1, -1),
        (0.1, 0.1),
    ),
)
def test_set_cct(sw, set_, expect):
    sw.set_cct(set_)
    assert sw.get_setup().cct_seconds == expect


@mark.parametrize(
    "set_, expect",
    (
        ((50, 300, 8), (50, 300, 8)),
        ((50, None, 8), (50, None, 8)),  # highpass
        ((50, 0, 8), (50, None, 8)),  # highpass
        ((None, 300, 8), (None, 300, 8)),  # lowpass
        ((0, 300, 8), (None, 300, 8)),  # lowpass
        ((None, None, 8), (None, None, 0)),  # bypass
        ((0, 0, 8), (None, None, 0)),  # bypass
        ((50, 2000, 8), (50, None, 8)),  # lowpass freq > nyquist -> highpass
        ((50, 300, 3), (None, None, 0)),  # invalid order -> disable
        ((400, 300, 8), (None, None, 0)),  # invalid filter freqs -> disable
    ),
)
def test_set_filter(sw, set_, expect):
    sw.set_filter(
        highpass=set_[0] * 1e3 if set_[0] else None,
        lowpass=set_[1] * 1e3 if set_[1] else None,
        order=set_[2],
    )
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == (expect[0] * 1e3 if expect[0] else None)
    assert setup.filter_lowpass_hz == (expect[1] * 1e3 if expect[1] else None)
    assert setup.filter_order == expect[2]


def test_set_datetime(sw):
    timestamp = datetime(2020, 12, 17, 18, 12, 33)
    sw.set_datetime(timestamp)
    assert sw.get_status().datetime == timestamp


@mark.parametrize(
    "set_, expect",
    (
        (-1, 0),
        (0, 0),
        (100, 100),
        (1_000_000, 1_000_000),
    ),
)
def test_threshold(sw, set_, expect):
    sw.set_threshold(set_)
    assert sw.get_setup().threshold_volts == expect / 1e6


@mark.parametrize(
    "set_, expect",
    (
        (False, False),
        (True, True),
    ),
)
def test_set_logging_mode(sw, set_, expect):
    sw.clear_data_log();  # logging can not be enabled if internal memory isn't cleared
    sw.set_logging_mode(set_)
    assert sw.get_setup().logging == expect
