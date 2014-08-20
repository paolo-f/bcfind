"""
Logging information to console
"""
from __future__ import print_function
import os.path
import sys
import traceback


class Colors:
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    magenta = '\033[95m'
    clean = '\033[0m'

FILTER = {'distance_filter','iterate'}


class Tee:
    """
    """
    def __init__(self):
        self.ostream = None
        self.logfilename = ''

    def logto(self, logfilename, mode='w'):
        if self.logfilename != '':
            self.ostream.close()
        self.logfilename = logfilename
        self.ostream = open(logfilename,mode)

    def error(self, exctype, value, trback, xframe=sys._getframe()):
        s = "%s: %s" % (xframe.f_code.co_filename, xframe.f_lineno) + " Exception: %s (%s)\n" % (exctype,value)
        print(Colors.red+s+Colors.clean)
        if self.ostream is not None:
            print(s, file=self.ostream)
            traceback.print_tb(trback,file=self.ostream)
            self.ostream.flush()

    def log(self, *args, **kwargs):
        if 'filter' in kwargs and kwargs['filter'] in FILTER:
            return
        xframe = sys._getframe(1)
        fileName = os.path.split(xframe.f_code.co_filename)[-1]
        lineno = xframe.f_lineno
        argstring = " ".join([str(arg) for arg in args])
        if 'end' in kwargs:
            print(argstring,end=kwargs['end'])
            if self.ostream is not None:
                print("%s:%d: %s" % (fileName, lineno, argstring),end=kwargs['end'],file=self.ostream)
        else:
            print(argstring)
            if self.ostream is not None:
                print("%s:%d: %s" % (fileName, lineno, argstring),file=self.ostream)
        sys.stdout.flush()
        if self.ostream is not None:
            self.ostream.flush()

tee = Tee()

_old_excepthook = sys.excepthook


def myexcepthook(exctype, value, trback):
    tee.error(exctype, value, trback)
    _old_excepthook(exctype, value, trback)
sys.excepthook = myexcepthook
