"""English, Malay, and the two mixed, which is how Malaysian and Singapore business text is often written."""
from pathlib import Path

from obi_lookout import ObiLookout

lookout = ObiLookout()
for name in ("english", "malay", "mixed"):
    text = Path(__file__).parent.parent.joinpath("samples", f"{name}.txt").read_text(encoding="utf-8")
    print(f"\n=== {name}")
    for span in lookout.detect(text):
        print(f"  {span.label:<24} {span.text}")
