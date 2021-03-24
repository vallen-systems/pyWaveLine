from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
from freezegun import freeze_time
from numpy.testing import assert_allclose
from serial import Serial, SerialException

from waveline import SpotWave
from waveline.spotwave import Setup

ADC_TO_VOLTS = 1.74e-6


@pytest.fixture(autouse=True)
def mock_spotwave_adc_factor():
    """
    Mock SpotWave._adc_to_volts method globally.

    SpotWave.__init__ will call get_setup() to get adc_to_volts.
    Return a constant value instead.
    """
    with patch("waveline.spotwave.SpotWave._get_adc_to_volts") as method:
        method.return_value = ADC_TO_VOLTS
        yield method


@pytest.fixture(autouse=True)
def mock_spotwave_check_firmware_version():
    """Mock SpotWave._check_firmware_version method globally."""
    with patch("waveline.spotwave.SpotWave._check_firmware_version") as method:
        method.return_value = None
        yield method


@pytest.fixture(name="serial_mock")
def mock_serial_port():
    serial_mock = Mock(spec=Serial)
    serial_mock.is_open = False
    return serial_mock


def test_init_serial(serial_mock):
    sw = SpotWave(serial_mock)
    assert sw._ser == serial_mock  # pylint: disable=protected-access
    serial_mock.open.assert_called()


def test_init_port():
    with pytest.raises(SerialException):
        SpotWave("invalid_port_id")


def test_init_invalid_type():
    with pytest.raises(ValueError):
        SpotWave(123)


def test_get_setup(serial_mock):
    sw = SpotWave(serial_mock)

    response = [
        b"recording=1\n",
        b"logging=0\n",
        b"adc2uv=1.74\n",
        b"cct=-0.5 s\n",
        b"filter=10.5-350 kHz, order 4\n",
        b"cont=0\n",
        b"thr=3162.5 uV\n",
        b"ddt=250  us\n",
        b"dummy line without value",
        b" status_interval = 1000 ms ",
        b"tr_enabled=1\n",
        b"tr_decimation=2\n",
        b"tr_pre_trig=100\n",
        b"tr_post_dur=100\n",
        b"tr_max_samples=2097152\n",
    ]
    serial_mock.readlines.return_value = response
    setup = sw.get_setup()
    serial_mock.write.assert_called_with(b"get_setup\n")
    assert setup == Setup(
        recording=True,
        logging=False,
        cont_enabled=False,
        adc_to_volts=1.74e-6,
        threshold_volts=3162.5e-6,
        ddt_seconds=250e-6,
        status_interval_seconds=1,
        filter_highpass_hz=10.5e3,
        filter_lowpass_hz=350e3,
        filter_order=4,
        tr_enabled=1,
        tr_decimation=2,
        tr_pretrigger_samples=100,
        tr_postduration_samples=100,
        cct_seconds=-0.5,
    )

    # test special filter outputs
    response[4] = b"filter=none-350 kHz, order 4\n"
    serial_mock.readlines.return_value = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == None
    assert setup.filter_lowpass_hz == 350_000
    assert setup.filter_order == 4

    response[4] = b"filter=10.5-none kHz, order 4\n"
    serial_mock.readlines.return_value = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == 10_500
    assert setup.filter_lowpass_hz == None
    assert setup.filter_order == 4

    response[4] = b"filter=none-none kHz, order 0\n"
    serial_mock.readlines.return_value = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == None
    assert setup.filter_lowpass_hz == None
    assert setup.filter_order == 0

    # empty response
    serial_mock.readlines.return_value = []
    with pytest.raises(RuntimeError):
        sw.get_setup()


def test_get_info(serial_mock):
    sw = SpotWave(serial_mock)

    response = [
        b"fw_version=00.21\n",
        b"type=spotWave\n",
        b"model=201\n",
        b"input_range=94 dBAE\n",
    ]
    serial_mock.readlines.return_value = response
    info = sw.get_info()
    serial_mock.write.assert_called_with(b"get_info\n")

    assert info.firmware_version == "00.21"
    assert info.type_ == "spotWave"
    assert info.model == "201"
    assert info.input_range_decibel == 94

    # empty response
    serial_mock.readlines.return_value = []
    with pytest.raises(RuntimeError):
        sw.get_info()


def test_get_status(serial_mock):
    sw = SpotWave(serial_mock)

    response = [
        b"temp=24 \xc2\xb0C\n",
        b"recording=0\n",
        b"logging=0\n",
        b"log_data_usage=13 sets (0.12 %)\n",
        b"date=2020-12-17 15:11:42.17\n",
    ]
    serial_mock.readlines.return_value = response
    status = sw.get_status()
    serial_mock.write.assert_called_with(b"get_status\n")

    assert status.temperature == 24
    assert status.recording == False
    assert status.logging == False
    assert status.log_data_usage == 13
    assert status.datetime == datetime(2020, 12, 17, 15, 11, 42, 170_000)

    # empty response
    serial_mock.readlines.return_value = []
    with pytest.raises(RuntimeError):
        sw.get_status()


@freeze_time("2022-11-11")
def test_commands_without_response(serial_mock):
    sw = SpotWave(serial_mock)

    def assert_write(expected):
        serial_mock.write.assert_called_with(expected)

    sw.set_continuous_mode(True)
    assert_write(b"set_acq cont 1\n")

    sw.set_logging_mode(True)
    assert_write(b"set_data_log enabled 1\n")
    sw.set_logging_mode(False)
    assert_write(b"set_data_log enabled 0\n")

    sw.set_ddt(400)
    assert_write(b"set_acq ddt 400\n")

    sw.set_status_interval(2.2)
    assert_write(b"set_acq status_interval 2200\n")

    sw.set_tr_enabled(True)
    assert_write(b"set_acq tr_enabled 1\n")

    sw.set_tr_decimation(10)
    assert_write(b"set_acq tr_decimation 10\n")

    sw.set_tr_pretrigger(200)
    assert_write(b"set_acq tr_pre_trig 200\n")

    sw.set_tr_postduration(0)
    assert_write(b"set_acq tr_post_dur 0\n")

    sw.set_cct(0.1, sync=False)
    assert_write(b"set_cct interval 0.1\n")
    sw.set_cct(-0.1, sync=False)
    assert_write(b"set_cct interval -0.1\n")
    sw.set_cct(0.1, sync=True)
    assert_write(b"set_cct interval -0.1\n")

    sw.set_filter(highpass=100_000, lowpass=300_000, order=6)
    assert_write(b"set_filter 100.0 300.0 6\n")

    sw.set_datetime(datetime(2020, 12, 16, 17, 55, 13))
    assert_write(b"set_datetime 2020-12-16 17:55:13\n")

    sw.set_datetime()  # datetime.now() -> frozen time
    assert_write(b"set_datetime 2022-11-11 00:00:00\n")

    sw.set_threshold(100)
    assert_write(b"set_acq thr 100\n")

    sw.start_acquisition()
    assert_write(b"start_acq\n")

    sw.stop_acquisition()
    assert_write(b"stop_acq\n")


def test_get_ae_data(serial_mock):
    sw = SpotWave(serial_mock)

    response = [
        b"S temp=27 T = 2010240 A=21 R=502689 D=2000000 C=0 E=38849818 TRAI=0 flags=0\n",
        b"H temp=27 T=3044759 A=3557 R=24 D=819 C=31 E=518280026 TRAI=1 flags=0\n",
        b"\n",
    ]

    serial_mock.readline.side_effect = response
    ae_data = sw.get_ae_data()
    ADC_TO_EU = ADC_TO_VOLTS ** 2 * 1e14 / 2e6

    # status record
    s = ae_data[0]
    assert s.type_ == "S"
    assert s.time == pytest.approx(2010240 / 2e6)
    assert s.amplitude == pytest.approx(21 * ADC_TO_VOLTS)
    assert s.rise_time == pytest.approx(502689 / 2e6)
    assert s.duration == pytest.approx(2000000 / 2e6)
    assert s.counts == 0
    assert s.energy == pytest.approx(38849818 * ADC_TO_EU)
    assert s.trai == 0
    assert s.flags == 0

    # hit record
    h = ae_data[1]
    assert h.type_ == "H"
    assert h.time == pytest.approx(3044759 / 2e6)
    assert h.amplitude == pytest.approx(3557 * ADC_TO_VOLTS)
    assert h.rise_time == pytest.approx(24 / 2e6)
    assert h.duration == pytest.approx(819 / 2e6)
    assert h.counts == 31
    assert h.energy == pytest.approx(518280026 * ADC_TO_EU)
    assert h.trai == 1
    assert h.flags == 0


@pytest.mark.parametrize("raw", (False, True))
def test_get_tr_data(serial_mock, raw):
    sw = SpotWave(serial_mock)

    lines = [
        b"TRAI = 1 T=43686000 NS=13\n",
        b"TRAI=2 T=43686983 NS=27\n",
        b"\n",
    ]
    data = [np.arange(samples, dtype=np.int16) for samples in (13, 27)]
    binary_data = [arr.tobytes() for arr in data]

    serial_mock.readline.side_effect = lines
    serial_mock.read.side_effect = binary_data

    tr_data = sw.get_tr_data(raw=raw)
    serial_mock.write.assert_called_with(b"get_tr_data\n")

    assert tr_data[0].trai == 1
    assert tr_data[0].time == pytest.approx(43686000 / 2e6)
    assert tr_data[0].samples == 13
    assert tr_data[0].raw == raw
    if raw:
        assert_allclose(tr_data[0].data, data[0])
    else:
        assert_allclose(tr_data[0].data, data[0] * ADC_TO_VOLTS)

    assert tr_data[1].trai == 2
    assert tr_data[1].time == pytest.approx(43686983 / 2e6)
    assert tr_data[1].samples == 27
    assert tr_data[1].raw == raw
    if raw:
        assert_allclose(tr_data[1].data, data[1])
    else:
        assert_allclose(tr_data[1].data, data[1] * ADC_TO_VOLTS)


@pytest.mark.parametrize("raw", (False, True))
@pytest.mark.parametrize(
    "samples",
    (
        32,
        128,
        65536,
    ),
)
def test_get_data(serial_mock, samples, raw):
    sw = SpotWave(serial_mock)

    mock_data = (2 ** 15 * np.random.randn(samples)).astype(np.int16)
    serial_mock.readline.return_value = f"NS={samples}\n".encode()
    serial_mock.read.return_value = mock_data.tobytes()

    data = sw.get_data(samples, raw=raw)
    serial_mock.write.assert_called_with(f"get_data {samples}\n".encode())
    serial_mock.read.assert_called_with(samples * 2)
    if raw:
        assert_allclose(data, mock_data)
    else:
        assert_allclose(data, mock_data * ADC_TO_VOLTS)
