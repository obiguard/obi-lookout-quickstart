# Security

## Reporting a vulnerability

Please report security problems **privately**, not in a public issue: use GitHub's *Security > Report a vulnerability* on this repository.

Examples of what to report: a way to make `server.py` leak request text, bypass its API key, or run code; or a credential committed here by mistake.

## Scope and expectations

- This repository is a quick-start guide and a small reference server for the [obi-lookout](https://huggingface.co/obiguard/obi-lookout) model. It is provided as is, with no support commitment.
- The server is meant to run on your own machine or network. If you expose it, set `OBI_API_KEY` and put it behind TLS (it speaks plain HTTP).
- obi-lookout is a detector and is **not a guarantee** that all sensitive data is found. A value it misses is not a vulnerability. Please use a normal issue for that, and do not paste real personal data into it.
