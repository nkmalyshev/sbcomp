import time


class Profiler(object):
    def __init__(self, proc_name=None):
        self.proc_name = proc_name or 'process'

    def __enter__(self):
        # log_info(f'{self.proc_name} started')
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if exc_type is not None:
            print(f'exc_type - {exc_type}, exc_val - {exc_val}, exc_tb - {exc_tb}')

        print(f'{self.proc_name} finished in {self.end_time - self.start_time:0.4f}s')
