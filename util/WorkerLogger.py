import csv
from datetime import datetime
import os


class WorkerLogger:
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/worker_logger_{timestamp}.csv"

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.filename = filename
        self._init_file()

    def _init_file(self):
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "worker_id", "reward", "steps", "done", "elapsed_time"])

    def log(self, episode, worker_id, reward, steps, done, elapsed_time):
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, worker_id, reward, steps, done, elapsed_time])
