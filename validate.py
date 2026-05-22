import sys
errors = []

from tools import REGISTRY, get_openai_tools, dispatch
tools = get_openai_tools()
print(f'[OK] {len(REGISTRY)} tools: {sorted(REGISTRY.keys())}')

valid_types = {'string', 'integer', 'number', 'boolean'}
for t in tools:
    fn = t['function']
    for pname, pdef in fn['parameters']['properties'].items():
        if pdef['type'] not in valid_types:
            errors.append(f"Bad type: {fn['name']}.{pname}={pdef['type']}")
if not errors:
    print('[OK] parameter types')

r = dispatch('nonexistent', {})
assert 'error' in r
print('[OK] dispatch unknown')

from stuck import StuckDetector
sd = StuckDetector()
for _ in range(4):
    sd.record('bash', {'command': 'ls'}, {'stdout': 'x', 'error': None})
ok, sid = sd.check()
assert ok and sid == 'S1', f'S1 fail: got {sid}'
print('[OK] S1')

sd.reset()
for _ in range(3):
    sd.record('bash', {'command': 'ls'}, {'error': 'fail'})
ok, sid = sd.check()
assert ok and sid == 'S2', f'S2 fail: got {sid}'
print('[OK] S2')

sd.reset()
for _ in range(5):
    sd.record('think', {'thought': 'hmm'}, {'recorded': True, 'error': None})
ok, sid = sd.check()
assert ok and sid == 'S3', f'S3 fail: got {sid}'
print('[OK] S3')

sd.reset()
for i in range(3):
    sd.record('bash', {'command': 'a'}, {'stdout': 'ra', 'error': None})
    sd.record('bash', {'command': 'b'}, {'stdout': 'rb', 'error': None})
ok, sid = sd.check()
assert ok and sid == 'S4', f'S4 fail: got {sid}'
print('[OK] S4')

from context import micro_compact, emergency_snip
msgs = [{'role': 'system', 'content': 'sys'}]
for i in range(15):
    msgs.append({'role': 'tool', 'tool_call_id': str(i), 'content': 'x' * 100})
out = micro_compact(msgs)
cleared = sum(1 for m in out if m.get('content') == '[old result cleared]')
assert cleared == 5, f'cleared={cleared}'
print(f'[OK] micro_compact cleared={cleared}')
snipped = emergency_snip(msgs)
assert snipped[0]['role'] == 'system' and len(snipped) <= 6
print(f'[OK] emergency_snip len={len(snipped)}')

from tools.bash import _clean_output
noisy = '\x1b[32mDownloading\x1b[0m\r100%\r Done'
c = _clean_output(noisy)
assert '\x1b' not in c and '\r' not in c
print(f'[OK] clean_output: {repr(c)}')

print()
if errors:
    print('FAILURES:')
    for e in errors:
        print(f'  FAIL: {e}')
    sys.exit(1)
else:
    print('All checks passed.')
