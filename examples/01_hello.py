"""The shortest useful program. The first run downloads the model (about 1.2 GB); later runs start from the cache."""
from obi_lookout import ObiLookout

lookout = ObiLookout()
text = "Hubungi Aisha binti Rahman di aisha@example.test atau +60 12-345 6789."

for span in lookout.detect(text):
    print(f"{span.label:<12} {span.text!r}  (confidence {span.score:.2f}, characters {span.start}-{span.end})")
