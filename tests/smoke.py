"""Runs the real model once and checks it finds what it should. Downloads about 1.2 GB the first time: python -m tests.smoke"""
from obi_lookout import ObiLookout

text = "Hubungi Aisha binti Rahman di aisha@example.test atau +60 12-345 6789."
found = {(s.label, s.text) for s in ObiLookout().detect(text)}
expected = {("person", "Aisha binti Rahman"), ("email", "aisha@example.test"), ("phone", "+60 12-345 6789")}
missing = expected - found
assert not missing, f"not found: {missing}; got {found}"
print("ok:", sorted(found))
