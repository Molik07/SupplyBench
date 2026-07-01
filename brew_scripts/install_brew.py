import pty
import os
import time

pid, fd = pty.fork()
if pid == 0:
    # Child process
    os.execl('/bin/bash', 'bash', '-c', '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
else:
    # Parent process
    buffer = ""
    while True:
        try:
            output = os.read(fd, 1024)
            if not output: break
            out_str = output.decode('utf-8', errors='ignore')
            print(out_str, end='', flush=True)
            buffer += out_str
            
            if "Password:" in buffer:
                os.write(fd, b"shivansh\n")
                buffer = ""
            elif "RETURN/ENTER" in buffer or "to abort" in buffer:
                os.write(fd, b"\n")
                buffer = ""
        except OSError:
            break
    os.waitpid(pid, 0)
