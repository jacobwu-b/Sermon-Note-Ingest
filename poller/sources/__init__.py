"""The registry of known church adapters, keyed by ``CHURCHES`` name."""

from __future__ import annotations

from poller.sources.base import PollResult, SermonItem, SourceAdapter
from poller.sources.gracepres import GracepresAdapter
from poller.sources.hillside import HillsideAdapter
from poller.sources.lakepointe import LakepointeAdapter
from poller.sources.menlo import MenloPodbeanAdapter
from poller.sources.north_point import NorthPointAdapter
from poller.sources.pbc import PbcAdapter
from poller.sources.westgate import WestgateAdapter

__all__ = [
    "ADAPTERS",
    "PollResult",
    "SermonItem",
    "SourceAdapter",
]

# A new church adds its adapter class here, keyed by its CHURCHES name.
ADAPTERS: dict[str, type[SourceAdapter]] = {
    "menlo": MenloPodbeanAdapter,
    "pbc": PbcAdapter,
    "north_point": NorthPointAdapter,
    "westgate": WestgateAdapter,
    "lakepointe": LakepointeAdapter,
    "hillside": HillsideAdapter,
    "gracepres": GracepresAdapter,
}
