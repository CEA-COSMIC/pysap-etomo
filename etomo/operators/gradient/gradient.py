"""
This module contains classes for defining gradients operators. Taken from
pysap-mri with minor changes (name of parameters (data_op), change of data
type to float).
"""
import numpy as np

from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


class GradAnalysis(GradBasic, PowerMethod):
    """
    Gradient operator for the data attachment term in the reconstruction
    process in analysis formulation
    """

    def __init__(self, data_op):
        GradBasic.__init__(self, np.array(0), data_op.op, data_op.adj_op)
        self.data_op = data_op
        PowerMethod.__init__(self, self.trans_op_op, self.data_op.shape,
                             data_type=data_op.dtype, auto_run=False)
        self.get_spec_rad(extra_factor=1.1)


class GradSynthesis(GradBasic, PowerMethod):
    """
    Gradient operator for the data attachment term in the reconstruction
    process in synthesis formulation
    """

    def __init__(self, linear_op, data_op):
        GradBasic.__init__(self, np.array(0), self._op_method,
                           self._trans_op_method)
        self.data_op = data_op
        self.linear_op = linear_op
        coef = linear_op.op(np.zeros(data_op.shape).astype(data_op.dtype))
        PowerMethod.__init__(self, self.trans_op_op, coef.shape,
                             data_type=data_op.dtype, auto_run=False)
        self.get_spec_rad(extra_factor=1.1)

    def _op_method(self, data, *args, **kwargs):
        # pylint: disable=unused-argument
        return self.data_op.op(self.linear_op.adj_op(data))

    def _trans_op_method(self, data, *args, **kwargs):
        # pylint: disable=unused-argument
        return self.linear_op.op(self.data_op.adj_op(data))
