
from itertools import permutations, combinations, product

def get_all_mapping(A, B):
    return [{a:b for a,b in zip(A,item)} for item in list(product(B,repeat=len(A)))]

import datetime
def get_now():
    return str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

def try_mkdir(path):
    try: os.mkdir(path)
    except: pass

from enum import Enum

##
# @class TextColors
# @brief color codes for terminal. use println to simply print colored message
class TextColors(Enum):
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def println(self, msg):
        print(self.value + str(msg) + self.ENDC.value)

import time
from threading import Thread, Event
import collections
import numpy as np
from .singleton import Singleton
import socket

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

def differentiate(X, dt):
    V = (X[1:]-X[:-1])/dt
    return np.concatenate([V, [V[-1]]], axis=0)

def integrate(V, dt, X0=0):
    return X0 + (np.cumsum(V, axis=0) - V) * dt

def interpolate_double(X):
    X_med = (X[:-1] + X[1:]) / 2
    return np.concatenate(
        [np.reshape(np.concatenate([X[:-1],X_med], axis=1), (-1, X.shape[1])),
        X[-1:]], axis=0)

##
# @brief matrix multiplication of last 2 dimensions
def matmul_md(A, B):
    return np.sum((np.expand_dims(A, axis=-1) * np.expand_dims(B, axis=-3)), axis=-2)


def get_mean_std(X, outlier_count=2):
    X_ex = [x[0] for x in sorted(zip(X,np.linalg.norm(X-np.mean(X, axis=0), axis=1)), key=lambda x: x[1])][:-outlier_count]
    return np.mean(X_ex, axis=0), np.std(X_ex, axis=0)


class DummyObject:
    def __init__(self):
        pass


##
# @class PeriodicTimer
# @brief    Creates a timer that can wait for periodic events.
# @remark   It is recommended to stop timer thread when its use is expired.
class PeriodicTimer:
    ##
    # @param period     period of the timer event, in secondes
    def __init__(self, period):
        self.period = period
        self.__tic = Event()
        self.__stop = Event()
        self.thread_periodic = Thread(target=self.__tic_loop)
        self.thread_periodic.daemon = True
        self.thread_periodic.start()

    def __tic_loop(self):
        while not self.__stop.wait(timeout=self.period):
            self.__tic.set()

    ##
    # @brief    wait for next timer event
    def wait(self):  # Just waiting full period makes too much threading delay - make shorter loop
        while not self.__tic.wait(self.period / 100):
            pass
        self.__tic.clear()

    ##
    # @brief    stop the timer thread
    def stop(self):
        self.__stop.set()

    def __del__(self):
        self.stop()

    def call_periodic(self, fun, N=None, timeout=None, args=[], kwargs={}):
        i_call = 0
        time_start = time.time()
        while True:
            self.wait()
            fun(*args, **kwargs)
            i_call += 1
            if N is not None and i_call > N:
                break
            if timeout is not None and time.time() - time_start > timeout:
                break

    def call_in_thread(self, fun, N=None, timeout=None, args=[], kwargs={}):
        kwargs_new = dict(fun=fun, N=N, timeout=timeout, **kwargs)
        t = Thread(target=self.call_periodic, args=args, kwargs=kwargs_new)
        t.daemon = True
        t.start()


##
# @class PeriodicIterator
# @brief create an iterator that makes periodic value returns.
class PeriodicIterator(PeriodicTimer):
    def __init__(self, item_list, period):
        PeriodicTimer.__init__(self, period)
        self.item_list = item_list
        self.item_itor = item_list.__iter__()

    def next(self):
        self.wait()
        try:
            return next(self.item_itor)
        except StopIteration as e:
            self.stop()
            raise (e)

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.item_list)

##
# @class    GlobalLogger
# @brief    A singleton logger to record data anywhere in the code.
class GlobalLogger(Singleton, dict):
    def __init__(self):
        pass


##
# @class    GlobalTimer
# @brief    A singleton timer to record timings anywhere in the code.
# @remark   Call GlobalTimer.instance() to get the singleton timer.
#           To see the recorded times, just print the timer: print(global_timer)
# @param    scale       scale of the timer compared to a second. For ms timer, 1000
# @param    timeunit    name of time unit for printing the log
# @param    stack   default value for "stack" in toc
class GlobalTimer(Singleton):
    def __init__(self, scale=1000, timeunit='ms', stack=False):
        self.reset(scale, timeunit, stack)

    ##
    # @brief    reset the timer.
    # @param    scale       scale of the timer compared to a second. For ms timer, 1000
    # @param    timeunit    name of time unit for printing the log
    # @param    stack   default value for "stack" in toc
    def reset(self, scale=1000, timeunit='ms', stack=False):
        self.stack = stack
        self.scale = scale
        self.timeunit = timeunit
        self.name_list = []
        self.ts_dict = {}
        self.time_dict = collections.defaultdict(lambda: 0)
        self.min_time_dict = collections.defaultdict(lambda: 1e10)
        self.max_time_dict = collections.defaultdict(lambda: 0)
        self.count_dict = collections.defaultdict(lambda: 0)
        self.timelist_dict = collections.defaultdict(list)
        self.switch(True)

    ##
    # @brief    switch for recording time. switch-off to prevent time recording for optimal performance
    def switch(self, onoff):
        self.__on = onoff

    ##
    # @brief    mark starting point of time record
    # @param    name    name of the section to record time.
    def tic(self, name):
        if self.__on:
            if name not in self.name_list:
                self.name_list.append(name)
            self.ts_dict[name] = time.time()

    ##
    # @brief    record the time passed from last call of tic with same name
    # @param    name    name of the section to record time
    # @param    stack   to stack each time duration to timelist_dict, set this value to True,
    #                   don't set this value to use default setting
    def toc(self, name, stack=None):
        if self.__on:
            dt = (time.time() - self.ts_dict[name]) * self.scale
            self.time_dict[name] = self.time_dict[name] + dt
            self.min_time_dict[name] = min(self.min_time_dict[name], dt)
            self.max_time_dict[name] = max(self.max_time_dict[name], dt)
            self.count_dict[name] = self.count_dict[name] + 1
            if stack or (stack is None and self.stack):
                self.timelist_dict[name].append(dt)
            return dt

    ##
    # @brief    get current time and estimated time arrival
    # @param    name    name of the section to record time
    # @param    current current index recommanded to start from 1
    # @param    end     last index
    # @return   (current time, eta)
    def eta(self, name, current, end):
        dt = self.toc(name, stack=False)
        return dt, (dt / current * end if current != 0 else 0)

    ##
    # @brief    record and start next timer in a line.
    def toctic(self, name_toc, name_tic, stack=None):
        dt = self.toc(name_toc, stack=stack)
        self.tic(name_tic)
        return dt

    ##
    # @brief you can just print the timer instance to see the record
    def __str__(self):
        strout = "" 
        names = self.name_list
        for name in names:
            strout += "{name}: \t{tot_T} {timeunit}/{tot_C} = {per_T} {timeunit} ({minT}/{maxT})\n".format(
                name=name, tot_T=np.round(np.sum(self.time_dict[name]),1), tot_C=self.count_dict[name],
                per_T= np.round(np.sum(self.time_dict[name])/self.count_dict[name], 1),
                timeunit=self.timeunit, minT=round(self.min_time_dict[name],3), maxT=round(self.max_time_dict[name],3)
            )
        return strout

    ##
    # @brief use "with timer:" to easily record duration of a code block
    def block(self, key, stack=None):
        return BlockTimer(self, key, stack=stack)

    def __enter__(self):
        self.tic("block")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc("block")

##
# @class    BlockTimer
# @brief    Wrapper class to record timing of a code block.
class BlockTimer:
    def __init__(self, gtimer, key, stack=None):
        self.gtimer, self.key, self.stack = gtimer, key, stack

    def __enter__(self):
        self.gtimer.tic(self.key)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gtimer.toc(self.key, stack=self.stack)
    
    
import os

def allow_korean_comment_in_dir(packge_dir):
    python_files = os.listdir(packge_dir)
    for filename in python_files:
        if '.py' in filename:
            file_path = os.path.join(packge_dir, filename)
            prepend_line(file_path, "# -*- coding: utf-8 -*- \n")
            
            
def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

from functools import wraps

def record_time(func):
    gtimer = GlobalTimer.instance()
    @wraps(func)
    def __wrapper(*args, **kwargs):
        gtimer.tic(func.__name__)
        res = func(*args, **kwargs)
        gtimer.toc(func.__name__)
        return res
    return __wrapper

import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def send_recv(sdict, host, port, buffer_len=1024):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((host, port))

    rdict = {}
    try:
        sjson = json.dumps(sdict, cls=NumpyEncoder, ensure_ascii = False)
        sbuff = sjson.encode()
        client_socket.sendall(sbuff)

        rjson = client_socket.recv(buffer_len)
        send_recv.rjson = rjson
        rdict = json.loads(rjson)
        send_recv.rdict = rdict
        if rdict is None:
            rdict = {}
        rdict = {str(k): v for k,v in rdict.items()}
    except Exception as e:
        print(e)
    finally:
        client_socket.close()
    return rdict

# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
#
# def createKF(dim_z, dt, P, R, Q, X0=None):
#     dim_x = dim_z*2
#     f = KalmanFilter (dim_x=dim_x, dim_z=dim_z)
#     f.x = np.array([0., 0.]*dim_z) if X0 is None else X0
#     f.F = np.identity(dim_x)  # state transition matrix
#     for i_F in range(dim_z):
#         f.F[i_F*2, i_F*2+1] = 1
#     f.H = np.identity(dim_x)[::2,:] # measurement function
#
#     f.P *= P #covariance matrix
#     f.R *= R # measurement noise
#     f.Q = np.identity(dim_x)
#     for i_Q in range(dim_z):
#         f.Q[i_Q*2:i_Q*2+2,i_Q*2:i_Q*2+2] = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
#     return f

import inspect

def divide_kwargs(kwargs, func1, func2):
    keys1 = inspect.getargspec(func1).args
    keys2 = inspect.getargspec(func2).args
    kwargs1 = {k:v for k,v in kwargs.items() if k in keys1}
    kwargs2 = {k:v for k,v in kwargs.items() if k in keys2}
    return kwargs1, kwargs2

def inspect_arguments(func):
    argspec = inspect.getargspec(func)
    defaults = argspec.defaults if argspec.defaults is not None else []
    len_kwargs = len(defaults)
    args = argspec.args[:-len_kwargs] if len_kwargs > 0 else argspec.args
    return args, {k:v for k,v in zip(argspec.args[-len_kwargs:], defaults)}

def CallHolder(caller, arg_keys, *args, **kwargs):
    def fun(*args_rt, **kwargs_rt):
        kwargs_rt.update({k:v for k,v in zip(arg_keys, args_rt) if k is not None})
        kwargs_rt.update(kwargs)
        return caller(*args, **kwargs_rt)
    fun.caller=caller
    fun.arg_keys=arg_keys
    fun.args=args
    fun.kwargs=kwargs
    return fun


##
#@ class dummy class to imitate multiprocess.Value
class SingleValue:
    def __init__(self, _type, _value):
        self.type = _type
        self.value = _value

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise (RuntimeError('Boolean value expected.'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise(RuntimeError('Boolean value expected.'))
        
def round_it_str(iterable, dec=3):
    dec_str="%.{}f".format(dec)
    return ",".join(map(lambda x:dec_str%x, iterable))

def str_num_it(strnum, deliminater=","):
    if deliminater in strnum:
        return map(float, strnum.split(deliminater))
    else:
        return None

def str2num_split_blank(string, dtype=float):
    return map(dtype, " ".join(string.split()).split(" "))

import pickle
def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data

def save_json(filename, data):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, cls=NumpyEncoder,indent=2)

def load_json(filename):
    with open(filename, "r") as st_json:
        st_python = json.load(st_json)
    return st_python

def read_file(filename):
    buffer = ""
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            buffer += line
            if not line: break
    return buffer

def list2dict(item_list, item_names):
    return {jname: jval for jname, jval in zip(item_names, item_list)}

def dict2list(item_dict, item_names):
    return [item_dict[jname] for jname in item_names]

class Logger:
    ERROR_FOLDER = "logs"
    def __init__(self, countout=3):
        self.count=0
        self.countout=countout
        try: os.mkdir(self.ERROR_FOLDER)
        except: pass

    def log(self, error, prefix="", print_now=True):
        if prefix != "":
            prefix += "_"
        if print_now: print(error)
        self.count += 1
        save_json(os.path.join(self.ERROR_FOLDER, prefix + get_now()), error)
        return self.count < self.countout

def sigmoid(x):
    return 1 / (1 +np.exp(-x))


def sign_positive_bias(Q):
    signQ = np.sign(Q)
    return signQ.astype(np.int) + (signQ==0).astype(np.int)


##
# @class DummyBlock
# @brief dummy for None instance for with phrase
class DummyBlock:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

##
# @brief copy file and replace substring
# @param line_callback function to be called when a line with string_from appeared
def copyfile_replace(file_from, file_to, string_from, string_to, line_callback=None):
    fin = open(file_from, "rt")
    #output file to write the result to
    fout = open(file_to, "wt")
    #for each line in the input file
    for line in fin:
        #read replace the string and write to output file
        if line_callback is not None and string_from in line:
            line_callback(line, string_from, string_to)
        fout.write(line.replace(string_from, string_to))
    #close input and output files
    fin.close()
    fout.close()

##
# @brief    print confusion matrix
# @remark   rows: ground truth, cols: prediction
def print_confusion_mat(GT, Res):
    TP = np.sum(np.logical_and(GT, Res))
    FN = np.sum(np.logical_and(GT, np.logical_not(Res)))
    FP = np.sum(np.logical_and(np.logical_not(GT), Res))
    TN = np.sum(np.logical_and(np.logical_not(GT), np.logical_not(Res)))
    N = TP + FN + FP + TN
    print("   {:>10} {:>10} {:>10}".format("PP", "PN", N))
    print("GP {:10} {:10} {:10.2%}".format(TP, FN, float(TP) / (TP + FN)))
    print("GN {:10} {:10} {:10.2%}".format(FP, TN, float(TN) / (FP + TN)))
    print(
        "AL {:10.2%} {:10.2%} {:10.2%}".format(float(TP) / (TP + FP), float(TN) / (TN + FN), float(TP + TN) / N))

def compare_dict(dict1, dict2):
    if sorted(dict1.keys()) != sorted(dict2.keys()):
        return False
    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, dict):
            res = compare_dict(val1, val2)
        else:
            res = val1 == val2
            if isinstance(res, list) or isinstance(res, np.ndarray):
                res = np.all(res)
        if not res:
            return res
    return True

def moving_median(values, window=3):
    assert window>1, "window should be larger than 1"
    assert window%2==1, "window should be odd number"
    n = window
    n1 = int((window-1)/2)
    values_med = (list(values[:n1]) 
                  + [np.median(values[i: i+n]) 
                     for i in range(len(values)-n+1)] 
                  + list(values[-n1:])
                 )
    return np.array(values_med)

##
# @brief convert ascii int list to string text
def ascii2str(ascii):
    return "".join(map(chr, ascii))

##
# @brief convert string text to ascii int list
def str2ascii(text):
    return map(ord, text)

##
# @brief get full name of given function including parent class
def fullname(fun):
    if hasattr(fun, '__qualname__'):
        return ".".join(fun.__qualname__.split(".")[-2:])
    elif hasattr(fun, 'im_class'):
        return "{}.{}".format(fun.im_class.__name__, fun.__name__)
    else:
        return fun.__name__

def extract_attr_dict(msg, valid_types=(int, float, str, tuple, list, dict)):
    if msg is None:
        return None
    if type(msg) in valid_types:
        if type(msg) == list:
            return [extract_attr_dict(submsg) for submsg in msg]
        return msg
    msg_dict = {}
    for k in dir(msg):
        if k.startswith('_'):
            continue
        submsg = getattr(msg, k)
        if callable(submsg):
            continue
        msg_dict[k] = extract_attr_dict(submsg)
    return msg_dict

from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

from collections import defaultdict

def swap_double_dict(double_dict):
    swapped = defaultdict(dict)
    for k1, v1 in double_dict.items():
        for k2, v2 in v1.items():
            swapped[k2][k1] = v2
    return swapped

def print_com_ports():
    import serial.tools.list_ports as sp
    com_list = sp.comports()
    connected = []
    for i in com_list:
        connected.append(i.device)

    print("Connected COM ports: " + str(connected))
