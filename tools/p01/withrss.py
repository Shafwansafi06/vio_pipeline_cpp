import os
import sys

cmd = sys.argv[1:]
if not cmd:
    sys.exit("usage: withrss.py CMD [ARGS...]")

pid = os.fork()
if pid == 0:
    try:
        os.execvp(cmd[0], cmd)
    except Exception as exc:
        os.write(2, ("exec failed: %s\n" % exc).encode())
        os._exit(127)

_, status, ru = os.wait4(pid, 0)
if os.WIFEXITED(status):
    code = os.WEXITSTATUS(status)
elif os.WIFSIGNALED(status):
    code = -os.WTERMSIG(status)
else:
    code = status
os.write(2, ("[RSS] maxrss_kb=%d\n[EXIT] %d\n" % (ru.ru_maxrss, code)).encode())
sys.exit(0)
