import psutil

process_name = "src.rl_runner"
processes = [p.cmdline() for p in psutil.process_iter() if len(p.cmdline()) > 2 and p.cmdline()[2] == process_name]
if len(processes) > 1:
    print("Master process up and running")
    exit()
print("No master process found - starting now")
