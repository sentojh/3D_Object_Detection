"""@package shared_function
This package is a helper to make function runs in a totally separatedshared process,        \n
just as single process.                                                                     \n
                                                                                            \n
###### EXAMPLE USAGE #####                                                                  \n
------------------ test_module.py -------------------                                       \n
from shared_function import shared_fun, CallType, ArgSpec, ResSpec, \                       \n
                      serve_forever                                                         \n
import subprocess                                                                           \n
                                                                                            \n
# decorate your function with shared_fun.                                                   \n
# you should setserver id for your function.                                                \n
# you should by CallType you can state this function will wait for response when called.    \n
# you should specify argument and result spec with ArgSpec and ResSpec.                     \n
@shared_fun("ServerID", CallType.SYNC,                                                      \n
    ArgSpec("kwarg1", (1,), float),                                                         \n
    ResSpec(1, (1,), bool))                                                                 \n
def positive(kwargA):                                                                       \n
    return [kwarg1[0]>0]                                                                    \n
                                                                                            \n
class TestClass:                                                                            \n
    @shared_fun("ServerID", CallType.SYNC,                                                  \n
        ArgSpec("kwarg1", (1,), float),                                                     \n
        ResSpec(1, (1,), bool))                                                             \n
    def negative(kwargA):                                                                   \n
        return [kwarg1[0]<0]                                                                \n
                                                                                            \n
if __name__ == "__main__":                                                                  \n
    tc = TestClass()                                                                        \n
    ## add server initialization on main so that created subprocess can start server        \n
    shared_function.serve_forever("ServerID", [positive, tc.negative])                      \n
else:                                                                                       \n
    ## create server subprocess when imported as module                                     \n
    subprocess.Popen(['python', 'test_module.py'])                                          \n
------------------------------------------------------                                      \n
                                                                                            \n
--------------- Your main procedure ------------------                                      \n
import test_module                                                                          \n
                                                                                            \n
## call function just as single process program, but must call with kwargs                  \n
res = postive(kwarg1=1)                                                                     \n
print(res)                                                                                  \n
tc = TestClass()                                                                            \n
res = tc.negative(kwarg1=-1)                                                                \n
print(res)                                                                                  \n
------------------------------------------------------                                      \n
"""

import SharedArray as sa
from decorator import decorator
from functools import wraps
from collections import namedtuple
from enum import Enum
import numpy as np
import copy
import time

##
# @brief flag to distinguish master and slave \n
#        set __SERVING = True to run function body, as in serve_forever
__SERVING = False

def set_serving(serving=True):
    global __SERVING
    __SERVING = serving

def is_serving():
    return __SERVING

##
# @brief shared function loop period is 1ms by default
SF_PERIOD = 1e-3

SHARED_FUN_URI_FORM = "shared_fun.{}.{}"

SHARED_FUNC_ALL = []

MAX_SHARE_SIZE = 100e6


def shared_fun_uri(addr, identifier):
    return SHARED_FUN_URI_FORM.format(addr, identifier)

##
# @class ArgSpec
# @brief properties for shared memory variable
# @param name   name of function argument
# @param shape tuple
# @param etype extended data type,
#              basic data type in SharedArray is any subtype of bool, int, float but not other object s.t. str
#              extended data type support str and json
ArgSpec = namedtuple("ArgSpec", ["name", "shape", "etype"])

##
# @class ResSpec
# @brief same as VarProb but for results
# @param index return value is distinguished by index
ResSpec = namedtuple("ResSpec", ["index", "shape", "etype"])


##
# @class CallType
# @brief flag for waiting for response (sync) or not (async)
class CallType(Enum):
    ASYNC = 0       # wait and return response
    SYNC = 1        # do not wait response, you can get response by check_response later


class ExtendedData:
    def __init__(self, addr, shape, etype):
        self.addr, self.shape, self.etype = addr, shape, etype
        if np.prod(self.shape)>MAX_SHARE_SIZE:
            print("[WARN] Data size too big for shared function: {} - {}".format(np.prod(self.shape), self.addr))
        self.dtype = dtype = self.dtype(etype)
        sm_dict = {sm.name.decode(): sm for sm in sa.list()}
        if addr in sm_dict:
            ex_shape, ex_dtype = sm_dict[addr].dims, sm_dict[addr].dtype
            if ex_shape == shape and ex_dtype == self.dtype:
                self.sm = sa.attach("shm://" + addr)
                return
            else:
                print("[WARN] SharedArray {} exists but property does not match: ({}, {}) <- ({},{})".format(
                        addr, ex_shape, ex_dtype, shape, dtype))
                sa.delete("shm://" + addr)
        self.sm = sa.create("shm://" + addr, shape=shape, dtype=dtype)


    def assign(self, data):
        if self.etype == dict:
            sjson = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)
            self.sm[:len(sjson)+1] = list(map(ord, sjson))+[0]
        elif self.etype == str:
            self.sm[:len(data)+1] = list(map(ord, data)) + [0]
        else:
            self.sm[:] = data

    def read(self):
        if self.etype == dict:
            data = list(map(chr, self.sm[:]))
            sjson = "".join(data[:np.where(self.sm[:] == 0)[0][0]])
            return json.loads(sjson)
        elif self.etype == str:
            data = list(map(chr, self.sm[:]))
            return "".join(data[:np.where(self.sm[:] == 0)[0][0]])
        else:
            return np.copy(self.sm)

    @classmethod
    def dtype(cls, etype):
        if etype == bool:
            return np.bool_
        elif etype == int:
            return np.int64
        elif etype == float:
            return np.float64
        elif etype == dict:
            return np.uint8
        elif etype == str:
            return np.uint8
        else:
            return etype

##
# @param ctype  CallType
# @param key    identifying key for shared memory
def shared_fun(ctype, key, *var_props):
    def __decorator(func):
        fullname_fun = ".".join([func.__name__, str(key).lower()])
        request = ExtendedData(shared_fun_uri(fullname_fun, "__request__"), shape=(1,), etype=bool)
        response = ExtendedData(shared_fun_uri(fullname_fun, "__response__"), shape=(1,), etype=bool)
        kwargs_s = {}
        returns_s = {}
        for var_prop in var_props:
            if isinstance(var_prop, ResSpec):
                addr = shared_fun_uri(fullname_fun, var_prop.index)
                returns_s[var_prop.index] = \
                    ExtendedData(addr, shape=var_prop.shape, etype=var_prop.etype)
            elif isinstance(var_prop, ArgSpec):
                addr = shared_fun_uri(fullname_fun, var_prop.name)
                kwargs_s[var_prop.name] = \
                    ExtendedData(addr, shape=var_prop.shape, etype=var_prop.etype)
            else:
                raise (TypeError("arguments for shared_fun should be either ArgSpec or ResSpec"
                                 "but {} is given".format(type(var_prop))))

        def __wrapper(self=None, timeout=None, **kwargs):
            if is_serving():
                res = __run_send_response(__wrapper, func, self)
            else:
                if ctype == CallType.ASYNC:
                    res = __send_request(__wrapper, **kwargs)
                elif ctype == CallType.SYNC:
                    res = __send_request_wait(__wrapper, timeout=timeout, **kwargs)
                else:
                    raise (TypeError("ctype should be CallType member"))
            return res

        __wrapper.ctype = ctype
        __wrapper.request = request
        __wrapper.response = response
        __wrapper.kwargs = kwargs_s
        __wrapper.returns = [returns_s[idx] for idx in sorted(returns_s.keys())]
        __wrapper.fullname = fullname_fun

        SHARED_FUNC_ALL.append(__wrapper)
        return __wrapper
    return __decorator

def __run_send_response(wrapepd, func, self):
    try:
        kwargs = {k: v.read() for k, v in wrapepd.kwargs.items()}
        if self is None:
            returns = func(**kwargs)
        else:
            returns = func(self, **kwargs)
        if returns is not None:
            if len(wrapepd.returns) == 1:
                wrapepd.returns[0].assign(returns)
            else:
                for edat, val in zip(wrapepd.returns, returns):
                    edat.assign(val)
        wrapepd.request.assign(False)
        wrapepd.response.assign(True)
    except Exception as e:
        error = get_error()
        error.assign("[ERROR] executing function call {}\n{} ".format(wrapepd.fullname, e))
        print(error.read())
        wrapepd.request.assign(False)
        wrapepd.response.assign(False)
        returns = None
    return returns


def __send_request(wrapepd, **kwargs):
    for k, val in kwargs.items():
        wrapepd.kwargs[k].assign(val)
    wrapepd.response.assign(False)
    wrapepd.request.assign(True)


def __send_request_wait(wrapepd, timeout=None, **kwargs):
    __send_request(wrapepd, **kwargs)
    return check_response(wrapepd, wait=True, timeout=timeout)


##
# @breif    get response from shared program. \n
#           return None if not request is not finished and wait=False
def check_response(wrapepd, wait=False, timeout=None):
    if not wrapepd.request.read()[0]:
        return get_return(wrapepd)
    elif wait:
        time_start = time.time()
        while wrapepd.request.read()[0]:
            time.sleep(SF_PERIOD)
            if timeout is not None and time.time() - time_start > timeout:
                break
        if not wrapepd.request.read()[0]:
            return get_return(wrapepd)
        else:
            raise(RuntimeError("Timeout during getting response - check last error state by get_error().read()"))
    else:
        return None

def get_return(wrapepd):
    if wrapepd.response.read()[0]:
        if len(wrapepd.returns) > 0:
            res = map(ExtendedData.read, wrapepd.returns)
            if res is None or len(res) == 0:
                res = None
            elif len(res) == 1:
                res = res[0]
            return res
        else:
            return None
    else:
        raise(RuntimeError(get_error().read()))


##
# @brief check if a computation server is running with given server_id
def get_pairing(server_id="all"):
    pairing_uri = shared_fun_uri("server.{}".format(server_id), "paired")
    paired = ExtendedData(pairing_uri, (1,), bool)
    return paired

##
# @brief check if a computation server is running with given server_id
def get_error():
    error_uri = shared_fun_uri("global", "error")
    error = ExtendedData(error_uri, (10000,), str)
    return error

def super_clear_shared_memory(names=None):
    if names is None:
        names = []
        for desc in sa.list():
            names.append(desc.name)

    for name in names:
        try:
            sa.delete(name)
        except:
            pass

def clear_channels_on(server_name):
    server_name = server_name.lower()
    snames = [sm.name for sm in sa.list()]
    for sname in snames:
        sname_split = sname.split(".")
        if len(sname_split)>=2 and sname_split[-2]==server_name:
            try:
                sa.delete(sname)
            except Exception as e:
                print("Error on deleting shared memory {}".format(sname))
                print(e)

import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

##
# @brief serve forever
# @param shared_funs  list of all shared functions to be served
# @param server_id    to prevent duplicated execution, use unique and consistent server_id
def serve_forever(server_id, shared_funcs, daemon=True,
                  verbose=False):
    assert isinstance(server_id, str), "server_id should be string instance"
    server_id = server_id.lower()
    pairstate_onterm = not daemon
    if not is_serving():
        print("[WARN] set_serving(True) was not called before calling server_forever.")
        print("       If you have initiailization dependent on is_serving(), it may have not worked.")
    set_serving(True)
    try:
        pairing = get_pairing(server_id)
        error = get_error()
        if pairing.read()[0]:
            pairstate_onterm = True
            error.assign("[FATAL] Other server is already paired with {}. Shutting down this process".format(
                pairing.addr))
            raise(RuntimeError(error.read()))

        pairing.assign(True)
        print("=========================")
        print("[INFO] Start serving for:")
        for func in shared_funcs:
            print(" * {}".format(func.fullname))
        print("=========================")
        while pairing.read()[0]:
            for func in shared_funcs:
                if func.request.read()[0]:
                    if verbose:
                        print("[INFO] {} requested".format(func.fullname))
                    try:
                        func()
                        if verbose:
                            print("[INFO] {} returned".format(func.fullname))
                    except Exception as e:
                        error.assign("[ERROR] {} returned failed \n{}".format(func.fullname, e))
                        print(error.read())
            time.sleep(SF_PERIOD)
    except Exception as e:
        error.assign("[ERROR] server {} terminated unexpectedly \n{}".format(server_id, e))
        print(error.read())
    finally:
        pairing.assign(pairstate_onterm)
        time.sleep(0.1)