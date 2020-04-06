def format_time(s):
    h = s // (3600)
    s %= 3600
    m = s // 60
    s %= 60
    return "%02i:%02i:%02i" % (h, m, s)

class EmptyContextManager:
    
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass
