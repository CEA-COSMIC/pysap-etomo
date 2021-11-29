"""
This module contains usefull methods for electron tomography applications.
"""
import numpy as np
import matplotlib.pyplot as plt


def flatten(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of ndarray or ndarray
        the input dataset.

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of uplet
        the input list of array structure.
    """
    # Check input
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape = [x[0].shape]
    for data in x[1:]:
        y = np.concatenate((y, data.flatten()))
        shape.append(data.shape)

    return y, shape


def unflatten(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of uplet
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    offset = 0
    x = []
    for size in shape:
        start = offset
        stop = offset + np.prod(size)
        offset = stop
        x.append(y[start: stop].reshape(size))

    return x


def flatten_swtn(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """
    # Check input
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = []
    shape_dict = []
    for i in range(len(x)):
        dict_lvl = {}
        for key in x[i].keys():
            dict_lvl[key] = x[i][key].shape
            y = np.concatenate((y, x[i][key].flatten()))
        shape_dict.append(dict_lvl)

    return y, shape_dict


def unflatten_swtn(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    x = []
    offset = 0
    for i in range(len(shape)):
        sublevel = {}
        for key in shape[i].keys():
            start = offset
            stop = offset + np.prod(shape[i][key])
            offset = stop
            sublevel[key] = y[start: stop].reshape(shape[i][key])
        x.append(sublevel)
    return x


def flatten_wave(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """

    # Flatten the dataset
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape_dict = [x[0].shape]
    for x_i in x[1:]:
        dict_lvl = []
        for key in x_i.keys():
            dict_lvl.append((key, x_i[key].shape))
            y = np.concatenate((y, x_i[key].flatten()))
        shape_dict.append(dict_lvl)

    return y, shape_dict


def unflatten_wave(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    start = 0
    stop = np.prod(shape[0])
    x = [y[start:stop].reshape(shape[0])]
    offset = stop
    for shape_i in shape[1:]:
        sublevel = {}
        for key, value in shape_i:
            start = offset
            stop = offset + np.prod(value)
            offset = stop
            sublevel[key] = y[start: stop].reshape(value)
        x.append(sublevel)
    return x
