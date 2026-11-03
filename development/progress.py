"""Visible progress of long runs: each running task writes how many inner fits it has done.

A task (one process) counts every inner-CV fit through the common score of the four optimizers, plus the
final refit, and writes ``<workdir>/<level>/progress/<task>.json`` at most every ``WRITE_EVERY`` seconds. The
main process reads those files and prints a status line every ``STATUS_EVERY`` seconds. Only what is printed
changes: the score, the fits and what each task saves are the same.
"""
import json
import os
import time

WRITE_EVERY = 20             # seconds between writes of a task's progress file
STATUS_EVERY = 600           # seconds between status lines of the main process

_STATE = {'path': None}


def start(path, label, total):
    """Begin counting the fits of the task of this process."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _STATE.update(path=path, label=label, total=int(total), done=0, t0=time.time(), last=0.0)
    _write(force=True)


def tick(n=1):
    """One more fit done (no effect outside a task)."""
    if _STATE.get('path'):
        _STATE['done'] += n
        _write()


def finish():
    """Remove the progress file of the finished task."""
    path = _STATE.get('path')
    _STATE['path'] = None
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _write(force=False):
    now = time.time()
    if not force and now - _STATE['last'] < WRITE_EVERY:
        return
    _STATE['last'] = now
    tmp = _STATE['path'] + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'task': _STATE['label'], 'done': _STATE['done'], 'total': _STATE['total'],
                   'started': _STATE['t0'], 'updated': now}, f)
    os.replace(tmp, _STATE['path'])


def status_line(folder, finished, total, elapsed_min):
    """One line: finished tasks and, for each running task, fits done of the expected ones and its time."""
    running = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if not name.endswith('.json'):
                continue
            try:
                with open(os.path.join(folder, name), encoding='utf-8') as f:
                    s = json.load(f)
            except (OSError, ValueError):
                continue
            share = 100 * s['done'] / s['total'] if s['total'] else 0
            minutes = (time.time() - s['started']) / 60
            took = f'{minutes:.0f} min' if minutes < 90 else f'{minutes / 60:.1f} h'
            running.append(f"{s['task']} {s['done']}/{s['total']} fits ({share:.0f} %), {took}")
    head = f'[status {elapsed_min:6.1f} min] {finished}/{total} tasks finished'
    return head + ('; running: ' + ' | '.join(running) if running else '')
