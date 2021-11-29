"""
Heavily inspired by pysap-mri multichannel reconstructor.
"""
# System import
import warnings

# Third party import
from modopt.opt.linear import Identity

# Package import
from ..optimizers.utils.cost import GenericCost
from ..optimizers import condatvu, pogm, fista


class ReconstructorBase:
    """ This is the base reconstructor class for reconstruction.

    Notes
    -----
        For the Analysis case, finds the solution  for x of:
        ..math:: (1/2) * ||R x - y||^2_2 + mu * H (W x)

        For the Synthesis case, finds the solution of:
        ..math:: (1/2) * ||R Wt alpha - y||^2_2 + mu * H(alpha)

    Parameters
    ----------
    data_op: object of class Radon2D, Radon3D
        Defines the radon data operator R.
    linear_op: object of class LinearBase
        Defines the linear sparsifying operator W. This must operate on x and
        have 2 functions, op(x) and adj_op(coeff) which implements the direct
        adjoint operators.
    regularizer_op: operator, (optional default None)
        Defines the regularization operator for the regularization function H.
        If None, the  regularization chosen is Identity and the optimization
        turns to gradient descent.
    gradient_formulation: str between 'analysis' or 'synthesis',
        default 'synthesis'
        defines the formulation of the image model which defines the gradient.
    grad_class: Gradient class from operators.gradient.
        Points to the gradient class based on the gradient_formulation.
    init_gradient_op: bool, default True
        This parameter controls whether the gradient operator must be
        initialized right now.
        If set to false, the user needs to call initialize_gradient_op to
        initialize the gradient at right time before reconstruction
    verbose: int, optional default 0
        Verbosity levels
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
            30 => Print the debug information of operators if defined by class
            NOTE - High verbosity (>20) levels are computationally intensive.
    extra_grad_args: Extra Keyword arguments for gradient initialization
        This holds the initialization parameters used for gradient
        initialization which is obtained from 'grad_class'.
        Please refer to operators.gradient.base for reference.
        In case of sythesis formulation, the 'linear_op' is also passed as
        an extra arg
    """

    def __init__(self, data_op, linear_op, regularizer_op,
                 gradient_formulation, grad_class, init_gradient_op=True,
                 verbose=0, **extra_grad_args):
        self.data_op = data_op
        self.linear_op = linear_op
        self.prox_op = regularizer_op
        self.gradient_method = gradient_formulation
        self.grad_class = grad_class
        self.verbose = verbose
        self.extra_grad_args = extra_grad_args
        self.cost_op = None
        self.x_final = None
        self.y_final = None
        self.costs = None
        self.metrics = None
        if regularizer_op is None:
            warnings.warn("The prox_op is not set. Setting to identity. "
                          "Note that optimization is just a gradient descent.")
            self.prox_op = Identity()
        # TODO try to not use gradient_formulation and
        #  rely on static attributes
        # If the reconstruction formulation is synthesis,
        # we send the linear operator as well.
        if gradient_formulation == 'synthesis':
            self.extra_grad_args['linear_op'] = self.linear_op
        if init_gradient_op:
            self.initialize_gradient_op(**self.extra_grad_args)

    def initialize_gradient_op(self, **extra_args):
        """
        Initialize gradient operator and cost operators

        Parameters
        ----------
        extra_args:
            kwargs for GradAnalysis or GradSynthesis

        Returns
        -------
        None
        """
        self.gradient_op = self.grad_class(
            data_op=self.data_op,
            **extra_args,
        )

    def reconstruct(self, data, optimization_alg='pogm',
                    x_init=None, num_iterations=100, cost_op_kwargs=None,
                    **kwargs):
        """ This method calculates operator transform.

        Parameters
        ----------
        data: np.ndarray
            The acquired value in the data domain. This is y in above equation.
        optimization_alg: str (optional, default 'pogm')
            Type of optimization algorithm to use, 'pogm' | 'fista' |
            'condatvu'
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction. If None, the
            initialization will be zero
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        cost_op_kwargs: dict (optional, default None)
            specifies the extra keyword arguments for cost operations.
            please refer to modopt.opt.cost.costObj for details.
        kwargs: extra keyword arguments for modopt algorithm
            Please refer to corresponding ModOpt algorithm class for details.
            https://github.com/CEA-COSMIC/ModOpt/blob/master/\
            modopt/opt/algorithms.py
        """
        self.gradient_op.obs_data = data
        available_algorithms = ["condatvu", "fista", "pogm"]
        if optimization_alg not in available_algorithms:
            raise ValueError("The optimization_alg must be one of " +
                             str(available_algorithms))
        optimizer = eval(optimization_alg)
        if optimization_alg == "condatvu":
            kwargs["dual_regularizer"] = self.prox_op
            optimizer_type = 'primal_dual'
        else:
            kwargs["prox_op"] = self.prox_op
            optimizer_type = 'forward_backward'
        if cost_op_kwargs is None:
            cost_op_kwargs = {}
        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            linear_op=self.linear_op,
            verbose=self.verbose >= 20,
            optimizer_type=optimizer_type,
            **cost_op_kwargs,
        )
        self.x_final, self.costs, *metrics = optimizer(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=self.verbose,
                **kwargs)
        if optimization_alg == 'condatvu':
            self.metrics, self.y_final = metrics
        else:
            self.metrics = metrics[0]
        return self.x_final, self.costs, self.metrics
