import asyncio
import logging
from datetime import datetime

from waveline import LinWave

logging.basicConfig(level=logging.INFO)


async def channel_acquisition(stream, channel: int):
    async for time, data in stream:
        print(f"Channel {channel}, {time}, {len(data)} samples")


async def main():
    ip = LinWave.discover()[0]

    async with LinWave(ip) as lw:
        await lw.set_tr_decimation(channel=0, factor=10)  # 10 MHz / 10 = 1 MHz
        await lw.set_range(channel=0, range_volts=5)  # 5 V

        # create channel acquisition streams (wrapping async generators)
        streams = [
            channel_acquisition(
                lw.stream(channel, 1_000_000),
                channel,
            )
            for channel in (1, 2)
        ]

        try:
            await asyncio.gather(*streams, lw.start_acquisition())
        finally:
            await lw.stop_acquisition()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ...