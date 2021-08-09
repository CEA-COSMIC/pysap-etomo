"""
Example of a single image reconstruction
"""
from etomo.operators import Radon2D, WaveletPywt, HOTV
from etomo.reconstructors.forwardradon import RadonReconstructor
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np
import matplotlib.pyplot as plt


# Loading input data
image = get_sample_data('2d-mri')
img_size = image.shape[0]

# Create radon operator and simulate data
theta = np.arange(0., 180., 3.)
radon_op = Radon2D(angles=theta, img_size=img_size, gpu=True)
data = radon_op.op(image)

# Create operators
# linear_op = HOTV(img_shape=image.shape, order=1)
linear_op = WaveletPywt(wavelet_name='sym8', nb_scale=3)
regularizer_op = SparseThreshold(linear=linear_op, weights=3e-6)
reconstructor = RadonReconstructor(
    data_op=radon_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='analysis',
)

x_final, cost, *_ = reconstructor.reconstruct(
    data=data,
    optimization_alg='condatvu',
    num_iterations=200,
    cost_op_kwargs={'cost_interval': 5}
)

# Results
plt.plot(cost)
plt.yscale('log')
plt.show()

image_rec = pysap.Image(data=x_final)
image_rec.show()
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))