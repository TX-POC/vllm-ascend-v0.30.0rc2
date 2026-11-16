# SPDX-License-Identifier: Apache-2.0
"""Default-off passive host observations; no framework/device dependencies."""

from contextlib import suppress


def notify(observer, event, **fields):
    try:
        return observer.record(event, **fields)
    except Exception:
        with suppress(Exception):
            observer.capture.fail("passive request observation failed")
        return None


def observed_call(observer, name, operation, *args, context=None, **kwargs):
    token = notify(observer, "begin", name=name, context=context)
    try:
        result = operation(*args, **kwargs)
    except BaseException:
        notify(observer, "end", token=token, success=False)
        raise
    notify(observer, "end", token=token, success=True, result=result)
    return result


def job_context(job):
    return {
        name: getattr(job, name)
        for name in ("request_id", "generation", "layer", "first", "tokens", "final", "member", "ratio")
    }


class ObservedLock:
    def __init__(self, lock, observer, context):
        self.lock, self.observer, self.context = lock, observer, context

    def __enter__(self):
        token = notify(self.observer, "begin", name="D.runtime_lock_wait", context=self.context)
        try:
            result = self.lock.__enter__()
        except BaseException:
            notify(self.observer, "end", token=token, success=False)
            raise
        notify(self.observer, "end", token=token, success=True)
        return result

    def __exit__(self, *args):
        return self.lock.__exit__(*args)
