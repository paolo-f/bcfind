"""
Timer class (used to make time-measuring decorators)
"""
from __future__ import print_function
from functools import wraps
import time

class Timer(object):
    """Used to make time-measuring decorators"""
    def __init__(self,name):
        self.cumulative=0
        self.n_calls=0
        self.name=name

    def reset(self):
        """Reset the timer"""
        self.cumulative=0
        self.n_calls=0

    def timed(self,fun):
        """The decorator function"""
        @wraps(fun)
        def _wrapper(*args, **kwds):
            start = time.time()
            result = fun(*args, **kwds)
            elapsed = time.time() - start
            self.cumulative += elapsed
            self.n_calls += 1
            return result
        return _wrapper

    def __str__(self):
        if self.n_calls>0:
            time_per_call=self.cumulative/float(self.n_calls)
        else:
            time_per_call=0
        return 'Timer: {:<16} Cumulative: {:8.2f}s, #calls: {:3d}, {:8.2f}s/call'.format(self.name,self.cumulative,self.n_calls,time_per_call)

