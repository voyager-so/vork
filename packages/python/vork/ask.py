"""CPU questions use the same addresses as the command and session."""
import json
import os
from typing import Any, Mapping, Sequence, Union
from ._engine import run_engine


def ask(scene: Union[str, 'os.PathLike[str]'], questions: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Read an ordered batch of scene values without rendering."""
    result = run_engine(['ask', os.fspath(scene), '-'], input=json.dumps(list(questions), allow_nan=False),
                        text=True, encoding='utf-8', capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    answers = json.loads(result.stdout)
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise RuntimeError('vork ask returned an invalid answer list')
    return answers
