"""A kept-alive `vork session` worker: the GPU, compiled pipelines and the installed scene survive between renders."""
import json
import os
import subprocess
from typing import Any, Mapping, Optional, Sequence, Union

from ._call_keys import CALL_KEYS as _CALL_KEYS
from ._engine import engine as _engine

# The reply field carrying the reuse class (reused, updated, reinstalled or recompiled); named once so a rename is one edit.
_REUSE_FIELD = 'reuse'


class SessionError(RuntimeError):
    """A refused request; `kind` is usage, scene, device or io."""

    def __init__(self, kind: str, message: str):
        super().__init__(f'{kind}: {message}')
        self.kind = kind


class Session:
    """open a scene, render, close. To edit, open again with the whole edited scene;
    the worker compares it with the open one and reuses what it can. Requests run in order.

    open returns {'revision', 'reuse' (reused, updated, reinstalled or recompiled), 'changed'
    (authored track paths that differed; a number is an unnamed list position)}. render
    returns the render result (warnings, exposure) plus revision, reuse, installs and
    provisional. VORK_BIN selects the executable.
    """

    def __init__(self, *, gpu: Optional[Union[int, str]] = None):
        executable = _engine()
        args = [executable, 'session'] + (['--gpu', str(gpu)] if gpu is not None else [])
        try:
            self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             text=True, encoding='utf-8')
        except FileNotFoundError:
            raise RuntimeError(
                f"vork executable not found: {executable!r}. Add the directory containing "
                "vork to PATH (for example, export PATH=\"$HOME/.local/bin:$PATH\"), "
                "or set VORK_BIN to the full path to the vork executable.") from None
        self._next = 1
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self, scene: Union[str, 'os.PathLike[str]', Mapping[str, Any]], *, base: Optional[str] = None) -> dict:
        """Install the whole scene: a file path, or a scene dict whose relative files resolve against base
        (default the current directory). Omitted fields are defaults. Opening again replaces the document;
        a failed open keeps the previous revision. Returns {'revision', 'reuse', 'changed'}."""
        if isinstance(scene, Mapping):
            body: dict[str, Any] = {'op': 'open', 'document': dict(scene)}
            if base is not None:
                body['base'] = str(base)
        else:
            if base is not None:
                raise TypeError('base applies to a scene dict; a scene file resolves against its own directory')
            body = {'op': 'open', 'scene': str(scene)}
        reply = self._request(**body)
        return {'revision': reply['revision'], _REUSE_FIELD: reply[_REUSE_FIELD], 'changed': reply['changed']}

    def ask(self, questions: Sequence[Mapping[str, Any]]) -> list[Any]:
        """Read an ordered batch from the open scene without rendering."""
        reply = self._request(op='ask', questions=list(questions))
        answers = reply['answers']
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise RuntimeError('vork ask returned an invalid answer list')
        return answers

    def render(self, out: Union[str, dict, Sequence[Union[str, dict]]], *, at: Optional[Union[float, str]] = None,
               frame_start: Optional[int] = None, frame_end: Optional[int] = None,
               profile: Optional[str] = None, set: Optional[dict] = None,
               channels: Optional[Sequence[str]] = None, provisional: bool = False) -> dict:
        """Render the open document with today's render options. provisional=True returns the
        first picture available (one sample, no denoise), labelled; render again for the final."""
        body: dict[str, Any] = {'op': 'render', 'out': [p if isinstance(p, dict) else str(p) for p in ([out] if isinstance(out, (str, dict)) else out)]}
        for key, value in (set or {}).items():
            if key not in _CALL_KEYS:
                raise TypeError('unknown render-call field: ' + key)
        if at is not None:
            body['at'] = at
        if frame_start is not None:
            body['frame_start'] = frame_start
        if frame_end is not None:
            body['frame_end'] = frame_end
        if profile is not None:
            body['profile'] = str(profile)
        if set is not None:
            body['set'] = dict(set)
        if channels:
            body['channels'] = list(channels)
        if provisional:
            body['provisional'] = True
        reply = self._request(**body)
        return {'warnings': reply['result']['warnings'], 'exposure': reply['result']['exposure'],
                'revision': reply['revision'], _REUSE_FIELD: reply[_REUSE_FIELD], 'installs': reply['installs'],
                'provisional': reply.get('provisional') is True}

    def close(self) -> None:
        """Release the document and end the worker."""
        if self._closed:
            return
        self._closed = True
        try:
            self._request(op='close', _closing=True)
        finally:
            try:
                if self._process.stdin:
                    try:
                        self._process.stdin.close()
                    except BrokenPipeError:
                        pass  # The worker may already have exited.
            finally:
                try:
                    if self._process.stdout:
                        self._process.stdout.close()
                finally:
                    self._process.wait()

    def _request(self, _closing: bool = False, **body: Any) -> dict:
        if self._closed and not _closing:
            raise RuntimeError('vork session is closed')
        request_id = self._next
        message = json.dumps({'id': request_id, **body}, allow_nan=False) + '\n'
        self._next += 1
        try:
            self._process.stdin.write(message)
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except (BrokenPipeError, OSError, ValueError):
            line = ''
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"vork session ended (exit {code}) before replying to {body['op']}")
        reply = json.loads(line)
        if reply.get('id') != request_id:
            raise RuntimeError(f"vork session replied out of order (expected {request_id}, got {reply.get('id')})")
        if reply.get('ok') is not True:
            raise SessionError(str(reply.get('kind')), str(reply.get('error')))
        return reply
