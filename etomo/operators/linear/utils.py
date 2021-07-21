# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains usefull methods for electron tomography applications.
"""
import numpy as np
import scipy.fftpack as pfft
import matplotlib.pyplot as plt


def with_metaclass(meta, *bases):
    """
    Function from jinja2/_compat.py.

    License: BSD.

    Use it like this::

        class BaseForm(object):
            pass

        class FormType(type):
            pass

        class Form(with_metaclass(FormType, BaseForm)):
            pass

    This requires a bit of explanation: the basic idea is to make a
    dummy metaclass for one level of class instantiation that replaces
    itself with the actual metaclass.  Because of internal type checks
    we also need to make sure that we downgrade the custom metaclass
    for one level to something closer to type (that's why __call__ and
    __init__ comes back from type etc.).

    This has the advantage over six.with_metaclass of not introducing
    dummy classes into the final MRO.
    """

    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)

    return metaclass("temporary_class", None, {})


def monkeypatch(klass, methodname=None):
    """ Decorator extending class with the decorated callable.

    >>> class A:
    ...     pass
    >>> @monkeypatch(A)
    ... def meth(self):
    ...     return 12
    ...
    >>> a = A()
    >>> a.meth()
    12
    >>> @monkeypatch(A, 'foo')
    ... def meth(self):
    ...     return 12
    ...
    >>> a.foo()
    12

    Parameters
    ----------
    klass: class object
        the class to be decorated.
    methodname: str, default None
        the name of the decorated method. If None, use the function name.

    Returns
    -------
    decorator: callable
        the decorator.
    """

    def decorator(func):
        try:
            name = methodname or func.__name__
        except AttributeError:
            raise AttributeError(
                "{0} has no __name__ attribute: you should provide an "
                "explicit 'methodname'".format(func))
        setattr(klass, name, func)
        return func

    return decorator


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



def create_plots(operators, rec, cost=None, plot_cost=False, cost_interval=1,
                 sfn='reconstruction'):
    """
    Create appropriate matplotlib Figures from reconstructed images and cost
    evolution. See carac_ex.tomography.ex_pyetomo.py for an example.

    Parameters
    ----------
    operators: array
        list of names of linear operators used for reconstructions
    rec: array
        list of reconstructed 2D images to show
    cost: array, optional
        list of lists of cost values associated with each operator
    plot_cost: bool, default False
        activation key to plot evolution of cost
    cost_interval: int, optional
        number of iterations between two computations of the cost function
        If  not given and plot_cost is True, set to 1
    sfn: string, optional
        name of the matplotlib figure

    Returns
    -------
    figs: Matplotlib figures
        reconstructed images and cost evolution if asked
    """
    # Create enough subplots to show everything
    n_op = len(operators)
    n = int(np.ceil(np.sqrt(n_op)))
    fig, axis = plt.subplots(int(np.ceil(n_op / n)), n, figsize=(15, 10))
    fig.sfn = sfn

    # Show images
    if n_op == 1:
        img = axis.imshow(rec[0], cmap=plt.cm.gray)
        axis.set_title(operators[0], fontsize=20)
        fig.colorbar(img, ax=axis)
    else:
        for i, ax in enumerate(axis.flatten()):
            if i < len(rec):
                img = ax.imshow(rec[i], cmap=plt.cm.gray)
                ax.set_title(operators[i], fontsize=20)
                fig.colorbar(img, ax=ax)
            # Don't show axis if no image is drawn
            else:
                ax.set_visible(False)

    # Plot cost evolution over iterations if given
    if plot_cost and cost is not None:
        fig_cost, axis = plt.subplots(int(np.ceil(n_op / n)), n,
                                      figsize=(15, 10))
        fig_cost.sfn = sfn + "_cost"
        interval = cost_interval * np.arange(len(cost[0]))

        if n_op == 1:
            axis.plot(interval, cost[0])
            axis.set_yscale('log')
            axis.set_title(operators[0], fontsize=20)
        else:
            for i, ax in enumerate(axis.flatten()):
                if i < len(cost):
                    ax.set_yscale('log')
                    ax.plot(interval, cost[i])
                    ax.set_title(operators[i], fontsize=20)
                # Don't show axis if no plot
                else:
                    ax.set_visible(False)
        return [fig, fig_cost]

    return [fig]
