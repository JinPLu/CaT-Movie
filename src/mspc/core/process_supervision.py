"""监督器只管理自己建立的进程组，组长退出不代表后代退出。"""
from contextlib import contextmanager
import os
import signal
from subprocess import Popen, PIPE
import time


class SupervisorTerminated(BaseException):
    """终止请求必须穿过会吞掉 InterruptedError 的 selector 等待。"""


@contextmanager
def termination_handler():
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SupervisorTerminated(f"received {signal.Signals(signum).name}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def live_group_members(pgid):
    """僵尸已退出，等待父进程回收；不能把它计为仍在执行的后代。"""
    with Popen(['ps', '-axo', 'pid=,pgid=,stat='], stdout=PIPE, text=True) as process:
        rows, _ = process.communicate()
        if process.returncode:
            raise RuntimeError('cannot inspect owned process group')
    return [int(pid) for pid, group, status in (line.split() for line in rows.splitlines())
            if (int(group) == pgid or int(pid) == pgid) and not status.startswith('Z')]


def stop_process_group(pid, grace_seconds=2.0):
    """TERM 宽限后 KILL，确认整个自有组没有活进程。兼顾 spawn 尚未 setsid。"""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            # multiprocessing spawn 在 worker 第一行 setsid 之前尚无自有组。
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not live_group_members(pid):
                return
            time.sleep(.02)
    raise RuntimeError(f'owned process group {pid} still has live members after SIGKILL')
