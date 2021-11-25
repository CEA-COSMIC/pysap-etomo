"""
Example of a single image reconstruction
"""
# Third party import
import pysap
from pysap.data import get_sample_data
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np
import matplotlib.pyplot as plt

from etomo.operators import Radon3D, WaveletPywt, HOTV_3D
from etomo.reconstructors.forwardtomo import TomoReconstructor

# Loading input data
image = get_sample_data('3d-pmri')
image.data = np.sqrt(np.sum(image.data**2, axis=0))
img_size, nb_slices = image.shape[0], image.shape[2]
image_astra = np.zeros((nb_slices, img_size, img_size))
for i in range(nb_slices):
    image_astra[i, :, :] = np.abs(image.data[:, :, i])
image = pysap.Image(data=image_astra)

# Create radon operator and simulate data
theta = np.arange(0., 180., 3.)
radon_op = Radon3D(angles=theta, img_size=img_size, nb_slices=nb_slices)
data = radon_op.op(image.data)

# Create operators: we give Identity to SparseThreshold as the inputs it will
# be given will already be in the linear operator's image space
linear_op = WaveletPywt(wavelet_name='bior4.4', nb_scale=3)

regularizer_op = SparseThreshold(linear=Identity(), weights=2e-8)
reconstructor = TomoReconstructor(
    data_op=radon_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=10
)

# Run reconstruction
x_final, cost, *_ = reconstructor.reconstruct(
    data=data,
    optimization_alg='pogm',
    num_iterations=100,
    cost_op_kwargs={'cost_interval': 3},
)

# Results
plt.plot(cost)
plt.yscale('log')
plt.show()

image_rec = pysap.Image(data=x_final)
image_rec.show()
