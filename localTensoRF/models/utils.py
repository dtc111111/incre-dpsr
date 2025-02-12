import inspect
import torch
import functools
import numpy as np
from .base_model import BaseModel
from collections import defaultdict


def get_pose_model(name):
    return get_class(name, __name__, BaseModel)

def get_class(mod_name, base_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
       the module named mod_name, child of base_path.
    """
    mod_path = "{}.{}".format("models", mod_name)
    mod = __import__(mod_path, fromlist=[''])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]

def torchify(func):
    """Extends to NumPy arrays a function written for PyTorch tensors.

    Converts input arrays to tensors and output tensors back to arrays.
    Supports hybrid inputs where some are arrays and others are tensors:
    - in this case all tensors should have the same device and float dtype;
    - the output is not converted.

    No data copy: tensors and arrays share the same underlying storage.

    Warning: kwargs are currently not supported when using jit.
    """
    # TODO: switch to  @torch.jit.unused when is_scripting will work
    @torch.jit.ignore
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        device = None
        dtype = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device_ = arg.device
                if device is not None and device != device_:
                    raise ValueError(
                        'Two input tensors have different devices: '
                        f'{device} and {device_}')
                device = device_
                if torch.is_floating_point(arg):
                    dtype_ = arg.dtype
                    if dtype is not None and dtype != dtype_:
                        raise ValueError(
                            'Two input tensors have different float dtypes: '
                            f'{dtype} and {dtype_}')
                    dtype = dtype_

        args_converted = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg).to(device)
                if torch.is_floating_point(arg):
                    arg = arg.to(dtype)
            args_converted.append(arg)

        rets = func(*args_converted, **kwargs)

        def convert_back(ret):
            if isinstance(ret, torch.Tensor):
                if device is None:  # no input was torch.Tensor
                    ret = ret.cpu().numpy()
            return ret

        # TODO: handle nested struct with map tensor
        if not isinstance(rets, tuple):
            rets = convert_back(rets)
        else:
            rets = tuple(convert_back(ret) for ret in rets)
        return rets

    # BUG: is_scripting does not work in 1.6 so wrapped is always called
    if torch.jit.is_scripting():
        return func
    else:
        return wrapped

def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)

def get_cktps(dir_):
    checkpoints = []
    for p in dir_.glob('checkpoint_*.tar'):
    #for p in dir_.glob('checkpoints.th'):
        checkpoints.append((0, p))
    return sorted(checkpoints)[-1][1]

def pack_lr_parameters(params, base_lr, lr_scaling):
    '''Pack each group of parameters with the respective scaled learning rate.
    '''
    filters, scales = tuple(zip(*[
        (n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info('Parameters with scaled learning rate:\n%s',
                {s: [n for n, _ in ps] for s, ps in scale2params.items()
                 if s != 1})
    lr_params = [{'lr': scale*base_lr, 'params': [p for _, p in ps]}
                 for scale, ps in scale2params.items()]
    return lr_params


import logging

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


def set_logging_debug(mode: bool):
    if mode:
        logger.setLevel(logging.DEBUG)