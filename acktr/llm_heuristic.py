import json
import os
import re
import urllib.request


class LLMHeuristicClient(object):
    def __init__(self, model='gpt-4o-mini', enabled=False):
        self.enabled = enabled
        self.model = model
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    def available(self):
        return self.enabled and bool(self.api_key)

    def generate(self, prompt):
        if not self.available():
            return None

        url = self.base_url.rstrip('/') + '/chat/completions'
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You generate ONLY pure Python function source code. '
                        'Do not output explanation.'
                    )
                },
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.4,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + self.api_key,
            },
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode('utf-8'))
        except Exception:
            return None

        try:
            text = body['choices'][0]['message']['content']
        except Exception:
            return None
        return self._extract_code(text)

    @staticmethod
    def _extract_code(text):
        block = re.findall(r'```python\\n([\\s\\S]*?)```', text)
        if block:
            return block[0].strip()
        block = re.findall(r'```([\\s\\S]*?)```', text)
        if block:
            return block[0].strip()
        return text.strip()
