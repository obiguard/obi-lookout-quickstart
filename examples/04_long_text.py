"""Long text. The model reads a few hundred words at a time, so `detect` cuts long text into overlapping windows for you.

Measured on the released model: one call on a 33,000-character text took 351 seconds and found about half of the values, while windows took 54 seconds and
found 91 to 93%. Here, the same helper is used on a 17,000-character file and the windows it used are printed.
"""
import time
from pathlib import Path

from obi_lookout import ObiLookout, windows

lookout = ObiLookout()
text = Path(__file__).parent.parent.joinpath("samples", "long.txt").read_text(encoding="utf-8")
print(f"{len(text):,} characters, cut into {len(windows(text, lookout.size, lookout.overlap))} windows of up to {lookout.size}")

started = time.time()
spans = lookout.detect(text)
print(f"found {len(spans)} values in {time.time() - started:.1f} s")
