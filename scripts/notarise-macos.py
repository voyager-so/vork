#!/usr/bin/env python3
"""Submit an existing signed archive; always record a digest-bound outcome."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notarise(folder):
    folder = Path(folder).resolve()
    result = dict(status='failed', reason='Notarisation did not complete', submission_id=None,
                  run_id=os.environ.get('GITHUB_RUN_ID'))
    secrets = [os.environ.get(k, '') for k in
               ('APPLE_ID', 'APPLE_APP_SPECIFIC_PASSWORD', 'APPLE_TEAM_ID')]

    def redact(text):
        for secret in secrets:
            if secret:
                text = text.replace(secret, '[REDACTED]')
        return text

    def command(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if p.returncode:
            raise RuntimeError(p.stderr or p.stdout or 'Command failed')
        return p.stdout

    def save():
        (folder / 'notary-result.json').write_text(json.dumps(result, indent=2) + '\n')

    save()  # Cancellation or runner termination must not leave a stale Accepted receipt.
    try:
        archive, = folder.glob('vork-*-darwin-arm64.tar.gz')
        proof = json.loads((folder / 'package-build.json').read_text())
        result['archive_sha256'] = digest(archive)
        result['source_run_id'] = proof['run_id']
        if result['archive_sha256'] != proof['archive_sha256']:
            raise RuntimeError('Archive sha256 differs from package-build.json')
        if not all(secrets):
            raise RuntimeError('Apple notarisation credentials are required')
        with tempfile.TemporaryDirectory() as temp:
            tree = Path(temp) / 'tree'
            tree.mkdir()
            command(['tar', '-xzf', str(archive), '-C', str(tree)])
            zip_path = str(Path(temp) / 'notarise.zip')
            command(['ditto', '-c', '-k', str(tree), zip_path])
            # Submit separately so a later wait/network failure retains the submission id.
            auth = ['--apple-id', secrets[0], '--password', secrets[1], '--team-id', secrets[2]]
            submitted = json.loads(command(['xcrun', 'notarytool', 'submit', zip_path,
                                            *auth, '--output-format', 'json']))
            result['submission_id'] = submitted['id']
            save()
            outcome = json.loads(command(['xcrun', 'notarytool', 'wait', submitted['id'],
                                          *auth, '--output-format', 'json']))
            result['status'] = outcome['status']
            result['reason'] = outcome.get('message', outcome['status'])
            if digest(archive) != result['archive_sha256']:
                raise RuntimeError('Archive changed during notarisation')
    except Exception as error:
        result.update(status='failed', reason=redact(str(error)))
    result['reason'] = redact(result['reason'])
    save()
    accepted = result['status'] == 'Accepted' and bool(result['submission_id'])
    label = 'Accepted notarisation' if accepted else 'NOT RELEASABLE: notarisation failed'
    print(label)
    if not accepted:
        print('::warning::NOT RELEASABLE: see notary-result.json and the job summary')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write('## ' + label + '\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n')
    return 0 if result['status'] == 'Accepted' and result['submission_id'] else 1


if __name__ == '__main__':
    sys.exit(notarise(sys.argv[1]))
