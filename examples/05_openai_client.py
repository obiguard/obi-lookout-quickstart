"""Call the Docker server with the standard `openai` package (pip install openai). Start the server first: docker compose -f docker/compose.yaml up

The server is a span detector, not a chat model: the "assistant" reply is a JSON document listing what was found. `obi_redact` is an extra field that also
returns the text with the values replaced.
"""
import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-unless-you-set-OBI_API_KEY")

reply = client.chat.completions.create(
    model="obi-lookout",
    messages=[{"role": "user", "content": "Hubungi Aisha binti Rahman di aisha@example.test atau +60 12-345 6789."}],
    extra_body={"obi_redact": "label"},
)
result = json.loads(reply.choices[0].message.content)
for entity in result["entities"]:
    print(entity["label"], entity["text"], entity["score"])
print(result["redacted"])
