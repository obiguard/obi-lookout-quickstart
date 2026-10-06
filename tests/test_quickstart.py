"""Logic tests with a stand-in model, so they run in a second with no download. The real model is checked by tests/smoke.py."""
import json
import threading
import unittest
import urllib.error
import urllib.request

import obi_lookout as o
import server


class FakeModel:
    """Finds every occurrence of a few known strings, as the real model's output shape: {"entities": {label: [{text,start,end,confidence}]}}."""

    def __init__(self, known, overshoot=0):
        self.known, self.overshoot = known, overshoot

    def extract_entities(self, text, labels, **kwargs):
        out = {label: [] for label in labels}
        for value, label in self.known.items():
            at = text.find(value)
            while at >= 0:
                out[label].append({"text": value, "start": at, "end": at + len(value) + self.overshoot, "confidence": 0.9})
                at = text.find(value, at + 1)
        return {"entities": out}


class HelperTest(unittest.TestCase):
    def test_short_text_is_one_window(self):
        self.assertEqual(o.windows("abc", 800, 300), [(0, 3)])

    def test_windows_cover_the_text_and_overlap(self):
        text = " ".join(f"word{i}" for i in range(600))
        wins = o.windows(text, 800, 300)
        self.assertEqual(wins[0][0], 0)
        self.assertEqual(wins[-1][1], len(text))
        for (a1, b1), (a2, b2) in zip(wins, wins[1:]):
            self.assertLess(a2, b1)  # neighbours overlap
            self.assertGreater(a2, a1)  # and always move forward

    def test_a_value_is_found_once_wherever_it_falls_in_long_text(self):
        filler = "lorem ipsum dolor sit amet " * 40
        for pad in range(0, 1200, 137):
            text = filler[:pad] + " mail a@x.test now " + filler
            found = o.ObiLookout(model=FakeModel({"a@x.test": "email"})).detect(text)
            self.assertEqual([(s.text, s.label) for s in found], [("a@x.test", "email")], pad)

    def test_ties_go_to_the_label_order(self):
        model = FakeModel({"db-01.corp": "financial_account"})
        model2 = FakeModel({"db-01.corp": "internal_host"})
        both = o.resolve_overlaps(
            o.ObiLookout(model=model)._window("see db-01.corp") + o.ObiLookout(model=model2)._window("see db-01.corp"))
        self.assertEqual([s.label for s in both], ["internal_host"])

    def test_a_span_ending_past_the_text_is_clamped(self):  # seen on the real model, 2026-10-07
        text = "token abc123.\n"
        found = o.ObiLookout(model=FakeModel({"abc123.\n": "secret"}, overshoot=1)).detect(text)
        self.assertEqual([(s.start, s.end) for s in found], [(6, 13)])

    def test_redact_styles(self):
        spans = [o.Span(6, 14, "email", 1.0, "a@x.test")]
        self.assertEqual(o.redact("mail: a@x.test!", spans), "mail: [EMAIL]!")
        self.assertEqual(o.redact("mail: a@x.test!", spans, "mask"), "mail: ********!")
        with self.assertRaises(ValueError):
            o.redact("x", spans, "nope")

    def test_bad_window_settings_are_refused(self):
        with self.assertRaises(ValueError):
            o.ObiLookout(model=FakeModel({}), size=800, overlap=100, margin=100)


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lookout = o.ObiLookout(model=FakeModel({"a@x.test": "email"}))
        cls.http = server.make_server(lookout, "127.0.0.1", 0, api_key="secret-key", limit=200)
        cls.base = f"http://127.0.0.1:{cls.http.server_address[1]}"
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()

    def call(self, path, body=None, key="secret-key"):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health_needs_no_key(self):
        self.assertEqual(self.call("/healthz", key=None), (200, {"status": "ok"}))

    def test_wrong_or_missing_key_is_refused(self):
        self.assertEqual(self.call("/v1/models", key=None)[0], 401)
        self.assertEqual(self.call("/v1/models", key="wrong")[0], 401)
        self.assertEqual(self.call("/v1/models")[0], 200)

    def test_a_non_ascii_key_is_refused_not_a_server_error(self):
        self.assertEqual(self.call("/v1/models", key="clé")[0], 401)

    def test_chat_completion_shape_and_redaction(self):
        status, body = self.call("/v1/chat/completions", {"model": "obi-lookout", "obi_redact": "label",
                                                          "messages": [{"role": "user", "content": "mail a@x.test now"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        result = json.loads(body["choices"][0]["message"]["content"])
        self.assertEqual(result["entities"][0]["label"], "email")
        self.assertEqual(result["redacted"], "mail [EMAIL] now")

    def test_detect_endpoint(self):
        status, body = self.call("/detect", {"text": "mail a@x.test now"})
        self.assertEqual((status, body["entities"][0]["text"]), (200, "a@x.test"))

    def test_errors_use_the_openai_shape(self):
        for path, payload, code in [("/detect", {"text": "  "}, 400), ("/detect", {"text": "x" * 201}, 413),
                                    ("/v1/chat/completions", {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}, 404),
                                    ("/v1/chat/completions", {"stream": True, "messages": []}, 400), ("/nope", {}, 404)]:
            status, body = self.call(path, payload)
            self.assertEqual(status, code, (path, payload))
            self.assertIn("message", body["error"])


if __name__ == "__main__":
    unittest.main()
