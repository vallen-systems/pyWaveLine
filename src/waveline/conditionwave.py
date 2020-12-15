"""
conditionWave
=============

.. autosummary::
    :toctree: generated

    ConditionWave
"""

import asyncio
import copy
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
import logging
import socket
from threading import Lock
from typing import List, Optional

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class FilterSettings:
    """Filter settings."""
    highpass: Optional[float]
    lowpass: Optional[float]
    order: int = 8


@dataclass
class ChannelSettings:
    """Channel settings."""
    range_volts: float
    decimation_factor: int
    filter_settings: FilterSettings


class _AcquisitionStatus:
    """Helper class read and parse status data on control port during acquisition."""

    def __init__(self, stream_reader: asyncio.StreamReader):
        self._reader = stream_reader
        self._task = None
        self._lock = Lock()
        self._temperature = 0
        self._buffersize = 0

    async def _read_acquisition_status(self):
        logger.debug("Start reading acquisition status")
        try:
            while True:
                line = await self._reader.readuntil(b'\n')  # raises IncompleteReadError on EOF
                line = line.decode("utf-8").rstrip()

                try:
                    key, value = line.split("=")
                except ValueError:
                    logger.warning(f"Can not parse acqusition status '{line}'")

                if key == "temp":
                    # logger.debug(f"Temperature = {value} °C")
                    with self._lock:
                        self._temperature = value
                elif key == "buffer_size":
                    # logger.debug(f"Buffer size = {value}")
                    with self._lock:
                        self._buffersize = value
                elif key == "error":
                    logging.error(f"Error during acquisition: {value}")
                else:
                    raise logger.warning(f"Unknown status key '{key}'")
        except asyncio.IncompleteReadError:
            logger.warning("No more acquisition status to read, quit task")
        except asyncio.CancelledError:
            logger.debug("Stop reading acquisition status")

    async def start(self):
        """Start async task."""
        self._task = asyncio.create_task(self._read_acquisition_status())

    async def stop(self):
        """Stop async task."""
        self._task.cancel()
        self._task = None

    def get_temperature(self):
        """Get system temperatur."""
        with self._lock:
            return self._temperature

    def get_buffersize(self):
        """Get current buffer size."""
        with self._lock:
            return self._buffersize


def require_connected(func):
    def check(obj: "ConditionWave"):
        if not obj.connected:
            raise ValueError("Device not connected")

    @wraps(func)
    async def async_wrapper(self: "ConditionWave", *args, **kwargs):
        check(self)
        return await func(self, *args, **kwargs)

    @wraps(func)
    def sync_wrapper(self: "ConditionWave", *args, **kwargs):
        check(self)
        return func(self, *args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class ConditionWave:
    """API for conditionWave device."""

    CHANNELS = (1, 2)
    MAX_SAMPLERATE = 10_000_000
    RANGES = {
        0.05: 0,  # 50 mV
        5.0: 1,  # 5 V
    }
    PORT = 5432
    DEFAULT_SETTINGS = ChannelSettings(
        range_volts=0.05,
        decimation_factor=1,
        filter_settings=FilterSettings(
            highpass=None,
            lowpass=None,
            order=8
        ),
    )

    def __init__(self, address: str):
        self._address = address
        self._reader = None
        self._writer = None
        self._settings = copy.deepcopy(self.DEFAULT_SETTINGS)
        self._connected = False
        self._daq_active = False
        self._daq_status: Optional[_AcquisitionStatus] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.close()

    @classmethod
    def discover(cls, timeout: float = 0.5) -> List[str]:
        """
        Discover conditionWave devices in network.

        Args:
            timeout: Timeout in seconds

        Returns:
            List of IP adresses
        """
        message = b"find"
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", cls.PORT))
        sock.sendto(message, ("<broadcast>", cls.PORT))

        def get_response(timeout=timeout):
            sock.settimeout(timeout)
            while True:
                try:
                    _, (ip, _) = sock.recvfrom(len(message))
                    yield ip
                except socket.timeout:
                    break

        return sorted(get_response())

    @property
    def connected(self) -> bool:
        """Check if connected to device."""
        return self._connected

    @property
    def input_range(self) -> float:
        """Input range in volts."""
        return self._settings.range_volts

    @property
    def decimation(self) -> int:
        """Decimation factor."""
        return self._settings.decimation_factor

    @property
    def filter_settings(self) -> FilterSettings:
        """Filter settings."""
        return copy.deepcopy(self._settings.filter_settings)

    async def connect(self):
        """Connect to device."""
        if self.connected:
            return

        logger.info(f"Open connection {self._address}:{self.PORT}...")
        self._reader, self._writer = await asyncio.open_connection(self._address, self.PORT)
        self._connected = True

        logger.info("Set default/saved settings...")
        await self.set_range(self.input_range)
        await self.set_decimation(self.decimation)
        await self.set_filter(
            self._settings.filter_settings.highpass,
            self._settings.filter_settings.lowpass,
            self._settings.filter_settings.order,
        )

    async def close(self):
        """Close connection."""
        if not self.connected:
            return
        try:
            if self._daq_active:
                await self.stop_acquisition()

            logger.info(f"Close connection {self._address}:{self.PORT}...")
            self._writer.close()
            await self._writer.wait_closed()
            self._connected = False
        except:  # pylint: disable=bare-except
            pass

    @require_connected
    async def _write(self, message):
        logger.debug("Write message: %s", message)
        self._writer.write(f"{message}\n".encode())  # type: ignore
        await self._writer.drain()

    @require_connected
    async def get_info(self) -> str:
        """Get device information."""
        logger.info("Get info...")
        await self._write("get_info")
        data = await self._reader.read(1000)  # type: ignore
        return data.decode()

    @require_connected
    async def set_range(self, range_volts: float):
        """
        Set input range.

        Args:
            range_volts: Input range in volts (0.05, 5)
        """
        try:
            range_index = self.RANGES[range_volts]
        except KeyError:
            raise ValueError(f"Invalid range. Possible values: {list(self.RANGES.keys())}")

        logger.info(f"Set range to {range_volts} V ({range_index})...")
        await self._write(f"set_adc_range 0 {range_index:d}")
        self._settings.range_volts = range_volts

    @require_connected
    async def set_decimation(self, factor: int):
        """
        Set decimation factor.

        Args:
            factor: Decimation factor [1, 500]
        """
        factor = int(factor)
        if not 1 <= factor <= 500:
            raise ValueError("Decimation factor must be in the range of [1, 500]")

        logger.info(f"Set decimation factor to {factor}...")
        await self._write(f"set_decimation 0 {factor:d}")
        self._settings.decimation_factor = factor

    @require_connected
    async def set_filter(
        self,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        order: int = 8,
    ):
        """
        Apply IIR filter settings.

        Default is bypass.

        Args:
            highpass: Highpass frequency in Hz
            lowpass: Lowpass frequency in Hz
            order: IIR filter order
        """
        def value_or(value: Optional[float], default_value: float):
            if value is None:
                return default_value
            return value

        if highpass is None and lowpass is None:
            logger.info("Set filter to bypass")
            await self._write("set_filter 0")
        else:
            highpass_khz = value_or(highpass, 0) / 1e3
            lowpass_khz = value_or(lowpass, self.MAX_SAMPLERATE) / 1e3

            logger.info(f"Set filter to {highpass_khz}-{lowpass_khz} kHz (order: {order})...")
            await self._write(f"set_filter 0 {highpass_khz} {lowpass_khz} {order}")

        self._settings.filter_settings.highpass = highpass
        self._settings.filter_settings.lowpass = lowpass
        self._settings.filter_settings.order = order

    @require_connected
    async def start_acquisition(self):
        """Start data acquisition."""
        if self._daq_active:
            return
        logger.info("Start data acquisition...")
        await self._write("start")
        self._daq_status = _AcquisitionStatus(self._reader)
        await self._daq_status.start()
        self._daq_active = True

    @require_connected
    async def stream(self, channel: int, blocksize: int):
        """
        Async generator to stream channel data.

        Args:
            channel: Channel number [0, 1]
            blocksize: Number of samples per block

        Yields:
            Tuple of datetime and numpy array (in volts)
        """
        if channel not in self.CHANNELS:
            raise ValueError(f"Channel must be in {self.CHANNELS}")
        if not self._daq_active:
            raise RuntimeError("Data acquisition not started")

        logger.info(
            (
                f"Start data acquisition on channel {channel} "
                f"(blocksize: {blocksize}, range: {self.input_range} V)"
            )
        )
        port = int(self.PORT + channel)
        blocksize_bits = int(blocksize * 2)  # 16 bit = 2 * 8 byte
        to_volts = float(self.input_range) / (2 ** 15)

        timestamp = None
        interval = timedelta(seconds=self.decimation * blocksize / self.MAX_SAMPLERATE)

        reader, writer = await asyncio.open_connection(self._address, port)
        while True:
            buffer = await reader.readexactly(blocksize_bits)
            if timestamp is None:
                timestamp = datetime.now()
            yield timestamp, np.frombuffer(buffer, dtype=np.int16).astype(np.float32) * to_volts
            timestamp += interval
        writer.close()

    @require_connected
    async def stop_acquisition(self):
        """Stop data acquisition."""
        if not self._daq_active:
            return
        logger.info("Stop data acquisition...")
        await self._write("stop")
        await self._daq_status.stop()
        self._daq_active = False

    @require_connected
    def get_temperature(self) -> Optional[int]:
        """Get current (only during acquisition) device temperature."""
        if not self._daq_active or self._daq_status is None:
            return None
        return self._daq_status.get_temperature()

    @require_connected
    def get_buffersize(self) -> int:
        """Get buffer size during acquisition in bytes."""
        if not self._daq_active or self._daq_status is None:
            return 0
        return self._daq_status.get_buffersize()

    def __del__(self):
        if self._writer:
            self._writer.close()
