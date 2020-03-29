import time

_since = time.time()  # sacred, don't touch :)

def reset_timer():
    global _since
    _since = time.time()

def now(precision=0):
    global _since
    time_elapsed = time.time() - _since
    time_tag = '[{:.0f} m {:.0f} s] '.format(
        time_elapsed // 60, time_elapsed % 60)
    
    return time_tag

