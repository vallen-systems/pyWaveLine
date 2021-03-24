"""
Module for spotWave device.

All device-related functions are exposed by the `SpotWave` class.
"""

import collections
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Union

import numpy as np
from serial import EIGHTBITS, Serial
from serial.tools import list_ports

logger = logging.getLogger(__name__)

# key = value pattern for ae/tr data
# fast(est) and simple, accept spaces around "="
# _KV_PATTERN = re.compile(br"(\w+)\s*=\s*(\S+)")
# accept words as keys w/o values; this seems next faster (incl. \S)?!
_KV_PATTERN = re.compile(br"([^\s=]+)(?:\s*=\s*(\S+))?")


@dataclass
class Info:
    """Device information."""

    firmware_version: str  #: Firmware version (major, minor)
    type_: str  #: Device type
    model: str  #: Model identifier
    input_range_decibel: int  #: Input range in dBAE


@dataclass
class Status:
    """Status information."""

    temperature: int  #: Device temperature in °C
    recording: bool  #: Flag if acquisition is active
    logging: bool  #: Flag if logging is active
    log_data_usage: int  #: Log buffer usage in sets
    datetime: datetime  #: Device datetime


@dataclass
class Setup:
    """Setup."""

    recording: bool  #: Flag if acquisition is active
    logging: bool  #: Flag if logging is active
    cont_enabled: bool  #: Flag if continuous mode is enabled
    adc_to_volts: float  #: Conversion factor from ADC values to volts
    threshold_volts: float  #: Threshold for hit-based acquisition in volts
    ddt_seconds: float  #: Duration discrimination time (DDT) in seconds
    status_interval_seconds: float  #: Status interval in seconds
    filter_highpass_hz: Optional[float]  #: Highpass frequency in Hz
    filter_lowpass_hz: Optional[float]  #: Lowpass frequency in Hz
    filter_order: int  #: Filter order
    tr_enabled: bool  #: Flag in transient data recording is enabled
    tr_decimation: int  #: Decimation factor for transient data
    tr_pretrigger_samples: int  #: Pre-trigger samples for transient data
    tr_postduration_samples: int  #: Post-duration samples for transient data
    cct_seconds: float  #: Coupling check transmitter (CCT) / pulser interval in seconds


@dataclass
class AERecord:
    """
    AE data record, either status or hit data.

    Todo:
        - Documentation or data type with available hit flags
    """

    type_: str  #: Record type (hit or status data)
    time: float  #: Time in seconds
    amplitude: float  #: Peak amplitude in volts
    rise_time: float  #: Rise time in seconds
    duration: float  #: Duration in seconds
    counts: int  #: Number of positive threshold crossings
    energy: float  #: Energy (EN 1330-9) in eu (1e-14 V²s)
    trai: int  #: Transient recorder index (key between `AERecord` and `TRRecord`)
    flags: int  #: Hit flags


@dataclass
class TRRecord:
    """Transient data record."""

    trai: int  #: Transient recorder index (key between `AERecord` and `TRRecord`)
    time: float  #: Time in seconds
    samples: int  #: Number of samples
    data: np.ndarray  #: Array of transient data in volts (or ADC values if `raw` is `True`)
    raw: bool = False  #: ADC values instead of user values (volts)


def _as_int(string, default: int = 0):
    """Return first sequence as int."""
    return int(string.strip().partition(" ")[0] or default)


def _as_float(string, default: float = 0.0):
    """Return first sequence as float."""
    return float(string.strip().partition(" ")[0] or default)


def _multiline_output_to_dict(lines: List[bytes]):
    """Helper function to parse output from get_info, get_status and get_setup."""
    return collections.defaultdict(
        str,
        [(k.strip(), v.strip()) for k, _, v in [line.decode().partition("=") for line in lines]],
    )


class SpotWave:
    """
    Interface for spotWave devices.

    The spotWave device is connected via USB and exposes a virtual serial port for communication.
    """

    VENDOR_ID = 8849  #: USB vendor id of Vallen Systeme GmbH
    PRODUCT_ID = 272  #: USB product id of SpotWave device
    CLOCK = 2_000_000  #: Internal clock in Hz
    # TICKS_TO_SEC = 1 / CLOCK  # precision lost...?
    _MIN_FIRMWARE_VERSION = "00.25"

    def __init__(self, port: Union[str, Serial]):
        """
        Initialize device.

        Args:
            port: Either the serial port id (e.g. "COM6") or a `serial.Serial` port instance.
                Use the method `discover` to get a list of ports with connected spotWave devices.
        Returns:
            Instance of `SpotWave`

        Example:
            There are two ways constructing and using the `SpotWave` class:

            1.  Without context manager and manually calling the `close` method afterwards:

                >>> sw = waveline.SpotWave("COM6")
                >>> print(sw.get_setup())
                >>> ...
                >>> sw.close()

            2.  Using the context manager:

                >>> with waveline.SpotWave("COM6") as sw:
                >>>     print(sw.get_setup())
                >>>     ...
        """
        if isinstance(port, str):
            self._ser = Serial(port=port)
        elif isinstance(port, Serial):
            self._ser = port
        else:
            raise ValueError("Either pass a port id or a Serial instance")

        self._ser.baudrate = 115200  # doesn't matter
        self._ser.bytesize = EIGHTBITS
        self._ser.timeout = 1  # seconds
        self._ser.exclusive = True

        self.connect()
        self._check_firmware_version()
        self.stop_acquisition()  # stop acquisition if running
        self._adc_to_volts = self._get_adc_to_volts()  # get and save adc conversion factor
        self._adc_to_eu = (self._adc_to_volts ** 2) * 1e14 / self.CLOCK

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _check_firmware_version(self):
        def get_version_tuple(version_string: str):
            return tuple((int(hx, base=16) for hx in version_string.split(".")))

        version = self.get_info().firmware_version
        if get_version_tuple(version) < get_version_tuple(self._MIN_FIRMWARE_VERSION):
            raise RuntimeError(
                f"Firmware version {version} < {self._MIN_FIRMWARE_VERSION}. Upgrade required."
            )

    def _get_adc_to_volts(self):
        return self.get_setup().adc_to_volts

    def connect(self):
        """
        Open serial connection to the device.

        The `connect` method is automatically called in the constructor.
        You only need to call the method to reopen the connection after calling `close`.
        """
        if not self.connected:
            self._ser.open()

    def close(self):
        """Close serial connection to the device."""
        if not hasattr(self, "_ser"):
            return
        if self.connected:
            self._ser.close()

    @property
    def connected(self) -> bool:
        """Check if the connection to the device is open."""
        return self._ser.is_open

    @contextmanager
    def _timeout_context(self, timeout_seconds: float):
        """Temporary set serial read/write timeout within context."""
        old_timeout = self._ser.timeout
        self._ser.timeout = timeout_seconds
        yield None
        self._ser.timeout = old_timeout

    @classmethod
    def discover(cls) -> List[str]:
        """
        Discover connected spotWave devices.

        Returns:
            List of port names
        """
        ports = list_ports.comports()
        ports_spotwave = filter(
            lambda p: (p.vid == cls.VENDOR_ID) & (p.pid == cls.PRODUCT_ID),
            ports,
        )
        return [port.name for port in ports_spotwave]

    def readlines(self):
        # read lines using short timeout
        with self._timeout_context(0.1):
            return self._ser.readlines()

    def clear_buffer(self):
        """Clear input and output buffer."""
        self.readlines()
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def _send_command(self, command: str):
        command_bytes = command.encode("utf-8") + b"\n"  # str -> bytes
        logger.debug("Send command: %a", command_bytes)
        self._ser.write(command_bytes)

    def get_info(self) -> Info:
        """
        Get device information.

        Returns:
            Dataclass with device information
        """
        self._send_command("get_info")
        lines = self.readlines()
        if not lines:
            raise RuntimeError("Could not get device information")

        info_dict = _multiline_output_to_dict(lines)
        return Info(
            firmware_version=info_dict["fw_version"],
            type_=info_dict["type"],
            model=info_dict["model"],
            input_range_decibel=_as_int(info_dict["input_range"]),
        )

    def get_setup(self) -> Setup:
        """
        Get setup.

        Returns:
            Dataclass with setup information
        """
        self._send_command("get_setup")
        lines = self.readlines()
        if not lines:
            raise RuntimeError("Could not get setup")

        setup_dict = _multiline_output_to_dict(lines)

        def get_filter_settings(string):
            """
            Parse special filter setting row.

            Example:
                10.5-350 kHz, order 4
                10.5-none kHz, order 4
                none-350 kHz, order 4
                none-none kHz, order 0
            """
            match = re.match(
                r"\s*(?P<hp>\S+)\s*-\s*(?P<lp>\S+)\s+.*o(rder)?\D*(?P<order>\d)",
                string,
                flags=re.IGNORECASE,
            )
            if not match:
                return None, None, 0

            def khz_or_none(k):
                try:
                    return 1e3 * float(match.group(k))
                except:  # pylint: disable=bare-except
                    return None

            return khz_or_none("hp"), khz_or_none("lp"), int(match.group("order"))

        filter_settings = get_filter_settings(setup_dict["filter"])

        return Setup(
            recording=_as_int(setup_dict["recording"]) == 1,
            logging=_as_int(setup_dict["logging"]) == 1,
            cont_enabled=_as_int(setup_dict["cont"]) == 1,
            adc_to_volts=_as_float(setup_dict["adc2uv"]) / 1e6,
            threshold_volts=_as_float(setup_dict["thr"]) / 1e6,
            ddt_seconds=_as_float(setup_dict["ddt"]) / 1e6,
            status_interval_seconds=_as_float(setup_dict["status_interval"]) / 1e3,
            filter_highpass_hz=filter_settings[0],
            filter_lowpass_hz=filter_settings[1],
            filter_order=filter_settings[2],
            tr_enabled=_as_int(setup_dict["tr_enabled"]) == 1,
            tr_decimation=_as_int(setup_dict["tr_decimation"]),
            tr_pretrigger_samples=_as_int(setup_dict["tr_pre_trig"]),
            tr_postduration_samples=_as_int(setup_dict["tr_post_dur"]),
            cct_seconds=_as_float(setup_dict["cct"]),
        )

    def get_status(self) -> Status:
        """
        Get status.

        Returns:
            Dataclass with status information
        """
        self._send_command("get_status")
        lines = self.readlines()
        if not lines:
            raise RuntimeError("Could not get status")

        def parse_datetime(string) -> datetime:
            """Parse datetime with any digit number of second fractions."""
            dt, _, fsec = string.partition(".")
            result = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            result += timedelta(seconds=int(fsec) / 10 ** len(fsec))
            return result

        status_dict = _multiline_output_to_dict(lines)
        return Status(
            temperature=_as_int(status_dict["temp"]),
            recording=_as_int(status_dict["recording"]) == 1,
            logging=_as_int(status_dict["logging"]) == 1,
            log_data_usage=_as_int(status_dict["log_data_usage"]),
            datetime=parse_datetime(status_dict["date"]),
        )

    def set_continuous_mode(self, enabled: bool):
        """
        Enable/disable continuous mode.

        Threshold will be ignored.
        The length of the records is determined by `ddt` with `set_ddt`.

        Note:
            The parameters for continuous mode with transient recording enabled (`set_tr_enabled`)
            have to be chosen with care - mainly the decimation factor (`set_tr_decimation`) and
            `ddt` (`set_ddt`). The internal buffer of the device can store up to ~200.000 samples.

            If the buffer is full, data records are lost.
            Small latencies in data polling can cause overflows and therefore data loss.
            One record should not exceed half the buffer size (~100.000 samples).
            25% of the buffer size (~50.000 samples) is a good starting point.
            The number of samples in a record is determined by `ddt` and the decimation factor `d`:
            :math:`n = ddt_{\\mu s} \\cdot f_s / d = ddt_{\\mu s} \\cdot 2 / d`
            :math:`\\implies ddt_{\\mu s} \\approx 50.000 \\cdot d / 2`

            On the other hand, if the number of samples is small, more hits are generated and the
            CPU load increases.

        Args:
            enabled: Set to `True` to enable continuous mode
        """
        self._send_command(f"set_acq cont {int(enabled)}")

    def set_ddt(self, microseconds: int):
        """
        Set duration discrimination time (DDT).

        Args:
            microseconds: DDT in µs
        """
        self._send_command(f"set_acq ddt {int(microseconds)}")

    def set_status_interval(self, seconds: int):
        """
        Set status interval.

        Args:
            seconds: Status interval in s
        """
        self._send_command(f"set_acq status_interval {int(seconds * 1e3)}")

    def set_tr_enabled(self, enabled: bool):
        """
        Enable/disable recording of transient data.

        Args:
            enabled: Set to `True` to enable transient data
        """
        self._send_command(f"set_acq tr_enabled {int(enabled)}")

    def set_tr_decimation(self, factor: int):
        """
        Set decimation factor of transient data.

        The sampling rate of transient data will be 2 MHz / `factor`.

        Args:
            factor: Decimation factor
        """
        self._send_command(f"set_acq tr_decimation {int(factor)}")

    def set_tr_pretrigger(self, samples: int):
        """
        Set pre-trigger samples for transient data.

        Args:
            samples: Pre-trigger samples
        """
        self._send_command(f"set_acq tr_pre_trig {int(samples)}")

    def set_tr_postduration(self, samples: int):
        """
        Set post-duration samples for transient data.

        Args:
            samples: Post-duration samples
        """
        self._send_command(f"set_acq tr_post_dur {int(samples)}")

    def set_cct(self, interval_seconds: int, sync: bool = False):
        """
        Set coupling check ransmitter (CCT) / pulser interval.

        The pulser amplitude is 3.3 V.

        Args:
            interval_seconds: Pulser interval in seconds
            sync: Synchronize the pulser with the first sample of the `get_data` command
        """
        if sync and interval_seconds > 0:
            interval_seconds *= -1
        self._send_command(f"set_cct interval {interval_seconds}")

    def set_filter(
        self,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        order: int = 4,
    ):
        """
        Set IIR filter frequencies and order.

        Args:
            highpass: Highpass frequency in Hz (`None` to disable highpass filter)
            lowpass: Lowpass frequency in Hz (`None` to disable lowpass filter)
            order: Filter order
        """
        if highpass is None:
            highpass = 0
        if lowpass is None:
            lowpass = 0.5 * self.CLOCK  # nyquist
        logger.info(f"Set filter to {highpass / 1e3}-{lowpass / 1e3} kHz (order: {order})...")
        self._send_command(f"set_filter {highpass / 1e3} {lowpass / 1e3} {int(order)}")

    def set_datetime(self, timestamp: Optional[datetime] = None):
        """
        Set current date and time.

        Args:
            timestamp: `datetime.datetime` object, current time if `None`
        """
        if not timestamp:
            timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self._send_command(f"set_datetime {timestamp_str}")

    def set_threshold(self, microvolts: float):
        """
        Set threshold for hit-based acquisition.

        Args:
            microvolts: Threshold in µV
        """
        self._send_command(f"set_acq thr {microvolts}")

    def set_logging_mode(self, enabled: bool):
        """
        Enable/disable data log mode.

        Args:
            enabled: Set to `True` to enable logging mode
        """
        self._send_command(f"set_data_log enabled {int(enabled)}")

    def start_acquisition(self):
        """Start acquisition."""
        logger.info("Start acquisition")
        self._send_command("start_acq")

    def stop_acquisition(self):
        """Stop acquisition."""
        logger.info("Stop acquisition")
        self._send_command("stop_acq")

    def _read_ae_data(self) -> List[AERecord]:
        records = []
        while True:
            line = self._ser.readline()
            if line == b"\n":  # last line is an empty new line
                break

            logger.debug(f"Received AE data: {line}")

            record_type = line[:1]
            # parse key-value pairs in line; default value: 0
            matches = collections.defaultdict(int, _KV_PATTERN.findall(line))

            if record_type in (b"H", b"S"):  # hit or status data
                record = AERecord(
                    type_=record_type.decode(),
                    time=int(matches[b"T"]) / self.CLOCK,
                    amplitude=int(matches[b"A"]) * self._adc_to_volts,
                    rise_time=int(matches[b"R"]) / self.CLOCK,
                    duration=int(matches[b"D"]) / self.CLOCK,
                    counts=int(matches[b"C"]),
                    energy=int(matches[b"E"]) * self._adc_to_eu,
                    trai=int(matches[b"TRAI"]),
                    flags=int(matches[b"flags"]),
                )
                records.append(record)
            elif record_type == b"R":  # marker record start
                ...
            else:
                logger.warning(f"Unknown AE data record: {line}")

        return records

    def get_ae_data(self) -> List[AERecord]:
        """
        Get AE data records.

        Todo:
            - Implement parsing of record start marker

        Returns:
            List of AE data records (either status or hit data)
        """
        self._send_command("get_ae_data")
        return self._read_ae_data()

    def get_tr_data(self, raw: bool = False) -> List[TRRecord]:
        """
        Get transient data records.

        Args:
            raw: Return TR amplitudes as ADC values if `True`, skip conversion to volts

        Returns:
            List of transient data records
        """
        self._send_command("get_tr_data")

        records = []
        while True:
            headerline = self._ser.readline()
            if headerline == b"\n":  # last line is an empty new line
                break

            # parse key-value pairs in line; default value: 0
            matches = collections.defaultdict(int, _KV_PATTERN.findall(headerline))
            trai = int(matches[b"TRAI"])
            time_ = int(matches[b"T"])
            samples = int(matches[b"NS"])

            data = np.frombuffer(self._ser.read(2 * samples), dtype=np.int16)
            if not raw:
                data = np.multiply(data, self._adc_to_volts, dtype=np.float32)

            if len(data) != samples:
                raise RuntimeError(
                    f"TR data samples ({len(data)}) do not match expected number ({samples})"
                )

            record = TRRecord(
                trai=trai,
                time=time_ / self.CLOCK,
                samples=samples,
                data=data,
                raw=raw,
            )
            records.append(record)

        return records

    def stream(self, raw: bool = False) -> Iterator[Union[AERecord, TRRecord]]:
        """
        High-level method to continuously acquire data.

        Args:
            raw: Return TR amplitudes as ADC values if `True`, skip conversion to volts

        Yields:
            AE and TR data records

        Example:
            >>> with waveline.SpotWave("COM6") as sw:
            >>>     # apply settings
            >>>     sw.set_ddt(400)
            >>>     for record in sw.stream():
            >>>         # do something with the data depending on the type
            >>>         if isinstance(record, waveline.spotwave.AERecord):
            >>>             ...
            >>>         if isinstance(record, waveline.spotwave.TRRecord):
            >>>             ...
        """
        self.start_acquisition()
        try:
            while True:
                t = time.monotonic()
                yield from self.get_ae_data()
                yield from self.get_tr_data(raw=raw)
                t = time.monotonic() - t
                # avoid brute load
                if t < 0.005:
                    time.sleep(0.01)
        finally:
            self.stop_acquisition()

    def get_data(self, samples: int, raw: bool = False) -> np.ndarray:
        """
        Read snapshot of transient data with maximum sampling rate (2 MHz).

        Args:
            samples: Number of samples to read
            raw: Return ADC values if `True`, skip conversion to volts

        Returns:
            Array with amplitudes in volts (or ADC values if `raw` is `True`)
        """
        samples = int(samples)
        self._send_command(f"get_data {samples}")
        _ = self._ser.readline()  # will return NS=<samples>
        adc_values = np.frombuffer(self._ser.read(2 * samples), dtype=np.int16)
        if raw:
            return adc_values
        return np.multiply(adc_values, self._adc_to_volts, dtype=np.float32)

    def get_data_log(self) -> List[AERecord]:
        """
        Get logged AE data records data from internal memory

        Returns:
            List of AE data records (either status or hit data)
        """
        self._send_command("get_data_log")
        return self._read_ae_data()

    def clear_data_log(self):
        """Clear logged data from internal memory."""
        self._send_command("clear_data_log")
