import tvm
import logging
from tvm import autotvm
import numpy as np
import sys

function = None
inputArg = None

@autotvm.template 
def GEMMAutoTVM(arg1, arg2, arg3, arg4):
    global function
    global inputArg
    print(arg1, arg2, arg3, arg4)
    
    args = tuple([arg1, arg2, arg3, arg4])
    ops, bufs = function(*args)
    s = tvm.create_schedule(ops)
    com_operation2 = s.stages[2]

    x = com_operation2.op.axis[1]
    y = com_operation2.op.axis[2]
    k = com_operation2.op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_knob("split_y", [4, 8, 16, 32])
    cfg.define_knob("split_x", [4, 8, 16, 32])
    cfg.define_knob("split_k", [4, 8])

    xo, xi = com_operation2.split(x, cfg["split_x"].val)
    yo, yi = com_operation2.split(y, cfg["split_y"].val)
    ko, ki = com_operation2.split(k, cfg["split_k"].val)
    com_operation2.reorder(xo, ko, yo, xi, ki, yi)
    
    return s, bufs

def auto_schedule(func, args):
    """Automatic scheduler
    
    Args:
    -----------------
    func: function object
        similar to batch_gemm function mentioned above
    args: tuple
        inputs to func
    -----------------
    Returns:
    s: tvm.schedule.Schedule
    bufs: list of tvm.tensor.Tensor
    """

    print(type(args), *args)
    global function
    global inputArg
    inputArg = args
    function = func

    # # for stage 2
    # for (1, 1024, 1024, 1024): 32, 32, 8 : 3.5
    # for (2, 512, 512, 512) : 32, 32, 8 : 2.8
    # for (8, 1024, 32, 1024) 
    #    32, 8(16), 8 : 4.9
    #    32, 32, 8 : 5.6
    print("hhhh")
    task = autotvm.task.create(GEMMAutoTVM, args=(args), target='llvm')
    print("iiii")
    
    print(task.config_space)
    
    # logging config (for printing tuning log to the screen)
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    
    # There are two steps for measuring a config: build and run.
    # By default, we use all CPU cores to compile program. Then measure them sequentially.
    # We measure 5 times and take average to reduce variance.
    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=3))
    
    # begin tuning, log records to file `matmul.log`
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=10,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('matmul.log')])
    
    with autotvm.apply_history_best('matmul.log'):
        with tvm.target.create("llvm"):
            s, arg_bufs = GEMMAutoTVM(args)
            return s, arg_bufs