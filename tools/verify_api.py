"""Exercise a variant's real API with synthetic logs; never read local conversations.

Run with an environment containing FastAPI and httpx:
    python tools/verify_api.py --source b_bots
On a segment branch, use --source .
"""
import argparse
import asyncio
import ast
import os
from pathlib import Path
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, type=Path)
args = parser.parse_args()
source = args.source.resolve()
for path in source.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
sys.path.insert(0, str(source))

import httpx

failures = []

async def run():
    import pi_api
    import logger
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=pi_api.app, raise_app_exceptions=False),
        base_url='http://test',
    ) as client:
        async def check(name, method, route, assertion, **kwargs):
            try:
                response = await client.request(method, route, **kwargs)
                assert assertion(response), f'HTTP {response.status_code}: {response.text[:180]}'
                print(f'PASS {name}')
            except Exception as error:
                failures.append(name)
                print(f'FAIL {name}: {error}')

        await check('empty dashboard', 'GET', '/generate',
                    lambda r: r.status_code == 200 and r.json()['conversations'] == [])
        logger.initialize_logging()
        logger.log_user_transcript('Synthetic user message')
        logger.log_ai_reply('Synthetic assistant response')
        # Test the API against the exact CSV format emitted by its own logger.
        await check('dashboard reads its own logs', 'GET', '/generate',
                    lambda r: r.status_code == 200 and any(
                        c['message'] == 'Synthetic assistant response'
                        for c in r.json()['conversations']))
        await check('conversation roles preserved', 'GET', '/generate',
                    lambda r: r.status_code == 200 and any(
                        c['message'] == 'Synthetic user message' and c['type'] == 'user'
                        for c in r.json()['conversations']))
        await check('browser CORS preflight', 'OPTIONS', '/generate',
                    lambda r: r.status_code == 200 and
                    r.headers.get('access-control-allow-origin') == 'http://localhost:5173',
                    headers={'Origin': 'http://localhost:5173',
                             'Access-Control-Request-Method': 'GET'})
        await check('dashboard message submission', 'POST', '/generate/api/mood-message',
                    lambda r: 200 <= r.status_code < 300,
                    json={'message': 'Synthetic message', 'timestamp': '2026-11-19T00:00:00Z'})

original = Path.cwd()
with tempfile.TemporaryDirectory(prefix='balance-api-') as temp:
    os.chdir(temp)
    for folder in ['logs', 'audio_pcm', 'output_realtime', 'audio_output']:
        Path(folder).mkdir()
    try:
        asyncio.run(run())
    finally:
        os.chdir(original)
print(f'{len(failures)} failed behavior checks')
sys.exit(bool(failures))
