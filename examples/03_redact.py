"""Replace what was found, so the text can be logged or sent on. `label` gives [EMAIL]; `mask` gives asterisks."""
from pathlib import Path

from obi_lookout import ObiLookout

lookout = ObiLookout()
text = Path(__file__).parent.parent.joinpath("samples", "malay.txt").read_text(encoding="utf-8")
print(lookout.redact(text, "label"))
print()
print(lookout.redact(text, "mask"))
