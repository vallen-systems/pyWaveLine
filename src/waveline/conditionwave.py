"""
conditionWave
=============

.. autosummary::
    :toctree: generated

    ConditionWave
"""

import asyncio
from datetime import datetime, timedelta
import logging
import socket
from typing import List, Optional

import numpy as np


logger = logging.getLogger(__name__)


class ConditionWave:
    """API for conditionWave device."""

    CHANNELS = (1, 2)
    MAX_SAMPLERATE = 10_000_000
    RANGES = {
        0.05: 0,  # 50 mV
        5.0: 1,  # 5 V
    }
    DEFAULT_RANGE = 0.05
    PORT = 5432

    def __init__(self, address: str):
        self._address = address
        self._reader = None
        self._writer = None
        self._range = self.DEFAULT_RANGE
        self._decimation = 1
        self._filter_highpass: Optional[float] = 0
        self._filter_lowpass: Optional[float] = self.MAX_SAMPLERATE
        self._filter_order = 8
        self._connected = False
        self._daq_active = False
        self._task_read_acquisition_status = None
        self._lock = asyncio.Lock()
        self._temperature = 0
        self._buffersize = 0

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
    def input_range(self) -> float:
        """Input range in volts."""
        return self._range

    @property
    def decimation(self) -> int:
        """Decimation factor."""
        return self._decimation

    async def connect(self):
        """Connect to device."""
        if self._connected:
            return
        logger.info(f"Open connection {self._address}:{self.PORT}...")
        self._reader, self._writer = await asyncio.open_connection(self._address, self.PORT)
        self._connected = True

    async def close(self):
        """Close connection."""
        if not self._connected:
            return
        try:
            logger.info(f"Close connection {self._address}:{self.PORT}...")
            self._writer.close()
            await self._writer.wait_closed()
            self._connected = False
        except:  # pylint: disable=bare-except
            pass

    async def _write(self, message):
        logger.debug("Write message: %s", message)
        self._writer.write(f"{message}\r".encode())
        await self._writer.drain()

    async def get_info(self):
        """Print info."""
        logger.info("Get info...")
        await self._write("get_info")
        data = await self._reader.read(1000)
        print(data.decode())

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
        self._range = range_volts

    async def set_decimation(self, factor: int):
        """
        Set decimation factor.

        Args:
            factor: Decimation factor [1, 16]
        """
        factor = int(factor)
        logger.info(f"Set decimation factor to {factor}...")
        await self._write(f"set_decimation 0 {factor:d}")
        self._decimation = factor

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
        self._filter_highpass = highpass
        self._filter_lowpass = lowpass
        self._filter_order = order

        def value_or(value: Optional[float], default_value: float):
            if value is None:
                return default_value
            return value

        await self._write(
            "set_filter 0 {highpass} {lowpass} {order}".format(
                highpass=value_or(highpass, 0) / 1e3,
                lowpass=value_or(lowpass, self.MAX_SAMPLERATE) / 1e3,
                order=order,
            )
        )

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
                    async with self._lock:
                        self._temperature = value
                elif key == "buffer_size":
                    # logger.debug(f"Buffer size = {value}")
                    async with self._lock:
                        self._buffersize = value
                elif key == "error":
                    logging.error(f"Error during acquisition: {value}")
                else:
                    raise logger.warning(f"Unknown status key '{key}'")
        except asyncio.IncompleteReadError:
            logger.warning("No more acquisition status to read, quit task")
        except asyncio.CancelledError:
            logger.debug("Stop reading acquisition status")

    async def get_temperature(self):
        """Get system temperatur."""
        async with self._lock:
            return self._temperature

    async def get_buffersize(self):
        """Get current buffer size."""
        async with self._lock:
            return self._buffersize

    async def start_acquisition(self):
        """Start data acquisition."""
        if self._daq_active:
            return
        logger.info("Start data acquisition...")
        await self._write("start")
        self._task_read_acquisition_status = asyncio.create_task(self._read_acquisition_status())
        self._daq_active = True

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
                f"(blocksize: {blocksize}, range: {self._range} V)"
            )
        )
        port = int(self.PORT + channel)
        blocksize_bits = int(blocksize * 2)  # 16 bit = 2 * 8 byte
        to_volts = float(self._range) / (2 ** 15)

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

    async def stop_acquisition(self):
        """Stop data acquisition."""
        if not self._daq_active:
            return
        logger.info("Stop data acquisition...")
        await self._write("stop")
        self._task_read_acquisition_status.cancel()
        self._task_read_acquisition_status = None
        self._daq_active = False

    def __del__(self):
        if self._writer:
            self._writer.close()