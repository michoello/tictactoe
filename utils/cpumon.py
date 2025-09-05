import psutil, time

for _ in range(5):
    print("CPU %:", psutil.cpu_percent(interval=1, percpu=True))
    print("Memory %:", psutil.virtual_memory().percent)
