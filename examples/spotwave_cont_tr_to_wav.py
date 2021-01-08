"""
Continuously stream transient data and save to wav files in chunks.
"""

import logging
import wave
from dataclasses import asdict
from datetime import datetime

import numpy as np

from waveline import SpotWave
from waveline.spotwave import AERecord, TRRecord
import threading, queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WavWriter:
    """Incremental WAV file writer."""

    def __init__(self, filename: str, samplerate: int):
        logger.info(f"Create new wav file: {filename}")
        self._file = wave.open(filename, "wb")
        self._file.setparams((1, 2, samplerate, 0, "NONE", ""))
    
    def __del__(self):
        self._file.close()

    def write(self, block: np.ndarray):
        assert block.itemsize == 2  # 16 bit
        self._file.writeframes(block)


def main(basename: str, seconds_per_file: float):
    def get_filename():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{basename}_{timestamp}.wav"

    port = SpotWave.discover()[0]
    print(port)

    with SpotWave(port) as sw:
        sw.set_continuous_mode(True)
        sw.set_cct(0)
        sw.set_status_interval(0)
        sw.set_tr_enabled(True)
        sw.set_tr_decimation(1)  # 2 MHz
        sw.set_ddt(10_000)  # 10 ms
        sw.set_filter(None, None)  # deactivate IIR filter

        setup = sw.get_setup()
        samplerate = sw.CLOCK / setup.tr_decimation
        chunks_per_file = int(seconds_per_file / setup.ddt_seconds)

        def async_write():
            chunks = 0
            writer = WavWriter(get_filename(), samplerate)
            while trqueue:
                try:
                    tr = trqueue.get(timeout=0.1)
                except:
                    continue
                writer.write(tr.data)
                chunks += 1
                if chunks >= chunks_per_file:
                    logger.info(f"{chunks_per_file} chunks acquired")
                    writer = WavWriter(get_filename(), samplerate)
                    chunks = 0

            print('Write finished')

        trqueue = queue.SimpleQueue()
        threading.Thread(target=async_write).start()
        try:
            for record in sw.stream(raw=True):  # return ADC values with enabled raw flag
                if isinstance(record, TRRecord):
                    trqueue.put(record)
                elif isinstance(record, AERecord):
                    if record.trai == 0:
                        logger.warning("Missing record(s)")
        finally:
            trqueue = None #flag to stop


if __name__ == "__main__":
    try:
        main("trtest", 5)
    except KeyboardInterrupt:
        ...
