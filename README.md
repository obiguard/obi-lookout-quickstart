# obi-lookout quick start

Run [obi-lookout](https://huggingface.co/obiguard/obi-lookout) on your own text in about five minutes.

obi-lookout finds personal and business-sensitive spans (names, ID numbers, phone numbers, emails, addresses, secrets, internal hostnames and more) in
English, Malay and mixed text, including Malaysian and Singapore formats. It returns character positions and labels, so you can mask or block the text
before it reaches a language model or a log. It is a detector, not a chat model.

> **It is not a guarantee.** It will miss some values and flag some that are not sensitive. Do not use it as your only control.
> Its main scores come from a synthetic test set, and on a small test of real public text it scored much lower (see [Know the limits](#know-the-limits)).

All names, numbers and organisations in the samples and examples are made up. Any match with a real person or company is coincidence.

## Three ways to run it

| You want | Use |
|---|---|
| Call it from Python | [Python](#python) |
| Scan a file from the terminal | [Command line](#command-line) |
| An HTTP endpoint that speaks the OpenAI API | [Docker server](#docker-server-openai-compatible) |
| Click and try, nothing to install | [Colab notebook](#colab-notebook) |

The first run downloads the model, about 1.2 GB. After that it runs offline. No text leaves your machine.

### Python

```bash
git clone https://github.com/obiguard/obi-lookout-quickstart && cd obi-lookout-quickstart
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python examples/01_hello.py
```

```
person       'Aisha binti Rahman'  (confidence 1.00, characters 8-26)
email        'aisha@example.test'  (confidence 1.00, characters 30-48)
phone        '+60 12-345 6789'  (confidence 1.00, characters 54-69)
```

Use it in your own code:

```python
from obi_lookout import ObiLookout

lookout = ObiLookout()                      # loads the model once; keep this object
spans = lookout.detect(text)                # non-overlapping spans, in order
for s in spans:
    print(s.label, s.text, s.start, s.end, s.score)

print(lookout.redact(text, "label"))        # "...call [PERSON] on [PHONE]..."
print(lookout.redact(text, "mask"))         # "...call ************ on **********..."
```

`requirements.txt` lists `torch` and `peft` even though the `gliner2` package does not declare them: it fails to import without them.
Versions are pinned to the ones the model was tested with (torch 2.14.1, gliner2 2.0.0, transformers 4.57.6). Tested on Python 3.11 and 3.12.

**You will see a tokenizer warning on load** ("incorrect regex pattern ... set `fix_mistral_regex=True`"). Ignore it. It is a false alarm for this model: that flag is a patch for Mistral tokenizers, and setting it here breaks tokenization (it turns spaces into unknown tokens). The model was trained and scored without it.

More: [`examples/`](examples) has English/Malay/mixed text, redaction, long text and an OpenAI-client call, all on synthetic text in [`samples/`](samples).

### Command line

```bash
python obi_lookout.py samples/malay.txt                    # list what was found
python obi_lookout.py --redact label samples/malay.txt     # print the text with values replaced
echo "Email aisha@example.test" | python obi_lookout.py --json
```

### Docker server (OpenAI-compatible)

```bash
docker compose -f docker/compose.yaml up --build
```

The build downloads the model into the image (about 4 GB in total, CPU only), so the container then runs with no network. It listens on
`127.0.0.1:8000`.

```bash
curl -s localhost:8000/detect -d '{"text": "Email aisha@example.test", "redact": "label"}'
```

Or with the standard `openai` package ([`examples/05_openai_client.py`](examples/05_openai_client.py)):

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="obi-lookout",
    messages=[{"role": "user", "content": "Email aisha@example.test"}],
    extra_body={"obi_redact": "label"},       # optional: also return the redacted text
)
result = json.loads(reply.choices[0].message.content)   # {"entities": [...], "redacted": "Email [EMAIL]"}
```

What to know about the server:

- It is a span detector behind an OpenAI-shaped API, so existing clients and gateways can call it. The "assistant" reply is always a JSON document of what was found. The last user message is the one scanned. Streaming is not supported.
- Endpoints: `GET /healthz`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /detect`.
- **Set `OBI_API_KEY`** (see the comment in `docker/compose.yaml`) before you expose it beyond your own machine. Clients then send `Authorization: Bearer <key>`.
- It handles one request at a time, accepts up to 20,000 characters per request, and never logs request text.
- On a CPU it takes roughly 1.6 seconds per 1,400 characters, after about half a minute to load.

### Colab notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obiguard/obi-lookout-quickstart/blob/main/notebook.ipynb)

Runs on Google's free CPU (checked on Colab on 2026-10-07). If an install step clashes with Colab's preinstalled packages, open an issue. **Use the sample text only: anything you paste into Colab goes to Google's servers**, which defeats the point of a tool that
finds sensitive data. For real text, use one of the local options above.

## What it finds

| Label | Examples |
|---|---|
| `person` | Aisha binti Rahman, Daniel Whitmore |
| `id_number` | Malaysian IC, Singapore NRIC |
| `phone` | `+60 12-345 6789`, `+65 9000 0000` |
| `email` | `name@company.test` |
| `address` | street addresses, with postcode and state |
| `date_of_birth` | `12 March 1987` |
| `financial_account` | bank account numbers |
| `financial_figure` | `RM 12,500.00`, `USD 48,250.00` |
| `organisation` | company and institution names |
| `client_or_project_name` | `Project Falcon`, client names in business text |
| `secret` | API keys and tokens |
| `internal_host` | `db-stg-02.corp.internal` |
| `medical_or_sensitive_category` | **no training or test examples. Do not rely on it.** |

## Long text

The model reads a few hundred words at a time. On a 33,000-character text, one call took 351 seconds and found about half of the values. Cutting the text
into overlapping 800-character windows took 54 seconds and found 91 to 93%. `ObiLookout.detect` does this for you, and distrusts a value that a window edge
cut in half. [`examples/04_long_text.py`](examples/04_long_text.py) shows it. If you call the model directly with `gliner2`, you need to do this yourself.

## Know the limits

- **Overall strict F1 is 0.927** (95% interval 0.919 to 0.934) on 1,569 documents written by a language model from fake data. Strict means start, end and label must all match. On text that contains nothing sensitive, 7.1% of documents had at least one false alarm.
- On values that do not appear in the training data, recall is 0.918 (0.975 on values that do). The fairer guide to new text is the lower one.
- The weakest labels are `organisation` (F1 0.761), `client_or_project_name` (0.836, recall 0.599 on unseen names) and `internal_host` (0.887). A client name is easily taken for an organisation, and the reverse.
- It is weaker on text that is not prose. In a separate test of 16 layouts, F1 was 0.883 overall, and lowest on shell sessions (0.500): for example `deploy@db-host.corp.internal` can be reported as an email, and public IPs as internal hosts. Tables and access logs are also weaker.
- **On real documents it scores much lower.** On 104 passages of real public text (Wikipedia, Singapore government procurement and company-register tables, open-source documentation), strict F1 on the nine labels shared with other open models was **0.46** (95% interval 0.37 to 0.56), against 0.31 for the next best open model and 0.30 for Presidio. On technical text such as documentation, configs and command output it scored 0.31, and 20 of 31 passages with nothing sensitive had a false alarm. It is weak on table rows (addresses, registration numbers, amounts) and flags product names, file names and version strings. The sample favours obi-lookout, and on the randomly drawn part Presidio scored higher. The labels were made by a language model, not people, and the test has no business documents (emails, invoices, contracts). Phone numbers, bank accounts and client or project names could not be tested on real text. Details are on the [model card](https://huggingface.co/obiguard/obi-lookout). Everything else above is synthetic text from the same family as the training data.
- Only English, Malay and mixed text were measured.

Full tables, how it was built, and the evaluation data (so you can reproduce the numbers) are on the [model card](https://huggingface.co/obiguard/obi-lookout) and the
[evaluation dataset](https://huggingface.co/datasets/obiguard/obi-lookout-eval).

## Pin or follow

[`obi_lookout.py`](obi_lookout.py) pins the model to the released commit (`REVISION`) so this guide cannot break when the model changes. Set it to `"main"` to follow
the latest release.

## Tests

```bash
python -m unittest tests.test_quickstart     # logic, no model needed, about a second
python -m tests.smoke                        # runs the real model once
```

## Problems and questions

Open an issue here. For a security problem, see [SECURITY.md](SECURITY.md) instead. Please do not paste real personal data into an issue: describe the kind of text and what went wrong, or use a made-up example.

## Licence

This repository is Apache-2.0 (see [LICENSE](LICENSE)). The model is Apache-2.0, fine-tuned from GLiNER2-PII over the MIT-licensed mDeBERTa-v3-base encoder; its
[model card](https://huggingface.co/obiguard/obi-lookout) lists the attributions.
