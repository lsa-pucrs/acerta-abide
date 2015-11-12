#!/usr/bin/env python
import os
import re
import time
import shlex
import struct
import platform
import subprocess
# from tabulate import tabulate

identifier = '(([a-zA-Z]_)?([a-zA-Z0-9_]*))'
replacement_field = '{' + identifier + '}'

class config_dict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def _count_replacements(d):
    count = 0
    for key in d:
        if isinstance(d[key], str):
            count = count + len(re.findall(replacement_field, d[key]))
    return count

def format_config(s, d):
    for k in d:
        s = s.replace('{' + k + '}', str(d[k]))
    return s

def nrangearg(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    if not m:
        raise ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))

def gpurangearg(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    if not m:
        m = re.match(r'(gpu\d+)(,(gpu\d+))*$', string)
        return [ x.strip() for x in m.group(0).split(',') ]
    else:
        start = m.group(1)
        end = m.group(2) or start
        return list(range(int(start,10), int(end,10)+1))
    # raise ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")


def compute_config(config1, config2):

    final = config_dict(config1)
    final.update(config2)

    # while True:
    #     replacements = _count_replacements(final)

    for key in final:
        if isinstance(final[key], str):
            try:
                final[key] = format_config(final[key], final)
            except Exception, e:
                pass
        # if replacements == _count_replacements(final):
        #     break

    return final

# def print_config(config):
#     k = config.keys()
#     k.sort()
#     print tabulate([ [x, config[x]] for x in k ])

def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def get_terminal_size():
    """ getTerminalSize()
     - get width and height of console
     - works on linux,os x,windows,cygwin(windows)
     originally retrieved from:
     https://gist.github.com/jtriley/1108174
     http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
    """
    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.startswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        print "default"
        tuple_xy = (80, 25)      # default value
    return tuple_xy

def _get_terminal_size_windows():
    try:
        from ctypes import windll, create_string_buffer
        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom,
             maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
        pass

def _get_terminal_size_tput():
    # get terminal width
    # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
    try:
        cols = int(subprocess.check_call(shlex.split('tput cols')))
        rows = int(subprocess.check_call(shlex.split('tput lines')))
        return (cols, rows)
    except:
        pass

def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            cr = struct.unpack('hh',
                               fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            return cr
        except:
            pass
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (os.environ['LINES'], os.environ['COLUMNS'])
        except:
            return None
    return int(cr[1]), int(cr[0])

def gpu_to_cpu(model):
    from pylearn2.utils import serial
    from theano import shared
    already_fixed = {}

    currently_fixing = []

    blacklist = ["im_class", "func_closure", "co_argcount", "co_cellvars", "func_code", "append", "capitalize", "im_self", "func_defaults", "func_name"]
    blacklisted_keys = ["bytearray", "IndexError", "isinstance", "copyright", "main"]

    postponed_fixes = []

    class Placeholder(object):
        def __init__(self, id_to_sub):
            self.id_to_sub = id_to_sub

    class FieldFixer(object):
        def __init__(self, obj, field, fixed_field):
            self.obj = obj
            self.field = field
            self.fixed_field = fixed_field

        def apply(self):
            obj = self.obj
            field = self.field
            fixed_field = already_fixed[self.fixed_field.id_to_sub]
            setattr(obj, field, fixed_field)

    def fix(obj, stacklevel=0):
        prefix = ''.join(['.']*stacklevel)
        oid = id(obj)
        canary_oid = oid
        if oid in already_fixed:
            return already_fixed[oid]
        if oid in currently_fixing:
            return Placeholder(oid)
        currently_fixing.append(oid)
        if hasattr(obj, 'set_value'):
            rval = shared(obj.get_value())
            obj.__getstate__ = None
        elif obj is None:
            rval = None
        elif isinstance(obj, list):
            rval = []
            for i, elem in enumerate(obj):
                fixed_elem = fix(elem, stacklevel + 2)
                if isinstance(fixed_elem, Placeholder):
                    raise NotImplementedError()
                rval.append(fixed_elem)
        elif isinstance(obj, dict):
            rval = obj
        elif isinstance(obj, tuple):
            rval = []
            for i, elem in enumerate(obj):
                fixed_elem = fix(elem, stacklevel + 2)
                if isinstance(fixed_elem, Placeholder):
                    raise NotImplementedError()
                rval.append(fixed_elem)
            rval = tuple(rval)
        elif isinstance(obj, (int, float, str)):
            rval = obj
        else:
            field_names = dir(obj)
            for field in field_names:
                if isinstance(getattr(obj, field), types.MethodType):
                    continue
                if field in blacklist or (field.startswith('__')):
                    continue
                updated_field = fix(getattr(obj, field), stacklevel + 2)
                if isinstance(updated_field, Placeholder):
                    postponed_fixes.append(FieldFixer(obj, field, updated_field))
                else:
                    try:
                        setattr(obj, field, updated_field)
                    except Exception as e:
                        print("Couldn't do that because of exception: "+str(e))
            rval = obj
        already_fixed[oid] = rval
        assert canary_oid == oid
        del currently_fixing[currently_fixing.index(oid)]
        return rval

    model = fix(model)

    assert len(currently_fixing) == 0

    for fixer in postponed_fixes:
        fixer.apply()

    return model


def run_multi(args, call, gpus=0, threads=1):

    # Do not pool if you don't have enough concurr
    # Keep it low
    threads = max(1, min(threads, args['concurr']))

    manager = multiprocessing.Manager()
    q = manager.Queue()

    # If you just want GPUs, use a number
    if type(gpus) == int:
        if gpus > 0:
            for i in xrange(0, threads/gpus):
                for g in gpus:
                    q.put('gpu' + str(g))

    # If you are using named GPUs, allocate a resource queue
    elif len(gpus) > 1:
        for i in xrange(0, threads/len(gpus)):
            for g in gpus:
                q.put(g)

    configs = []
    for concurr in xrange(args['concurr']):
        args_config = compute_config({'concurr': str(concurr)}, args)
        if q.qsize() > 0:
            args_config['queue'] = q
        configs.append(args_config)

    # If you are not multithreading, run it as our ancestors did
    if threads == 1:
        for config in configs:
            call(config)

    # Or allocate a pool for multithreading
    else:
        pool = multiprocessing.Pool(processes=threads)
        pool.map(call, configs)
        pool.close()
        pool.join()

if __name__ == "__main__":

    sizex, sizey = get_terminal_size()
    print  'width =', sizex, 'height =', sizey

    print gpurangearg('gpu0,gpu1,gpu2')
    print gpurangearg('1-7')
