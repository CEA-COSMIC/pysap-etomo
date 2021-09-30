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

from etomo.operators import Radon2D, WaveletPywt, HOTV
from etomo.reconstructors.forwardradon import TomoReconstructor

# Loading input data
image = get_sample_data('2d-mri')
img_size = image.shape[0]

# Create radon operator and simulate data
theta = np.arange(0., 180., 3.)
radon_op = Radon2D(angles=theta, img_size=img_size, gpu=True)
data = radon_op.op(image)

# Create operators: we give Identity to SparseThreshold as the inputs it will
# be given will already be in the linear operator's image space
TV = HOTV(img_shape=image.shape, order=1)
wavelet = WaveletPywt(wavelet_name='sym8', nb_scale=3)
linear_op = TV

regularizer_op = SparseThreshold(linear=Identity(), weights=2e-6)
reconstructor = TomoReconstructor(
    data_op=radon_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='analysis',
)

# Run reconstruction
x_final, cost, *_ = reconstructor.reconstruct(
    data=data,
    optimization_alg='condatvu',
    num_iterations=300,
    cost_op_kwargs={'cost_interval': 5}
)

# Results
plt.plot(cost)
plt.yscale('log')
plt.show()

image_rec = pysap.Image(data=x_final)
image_rec.show()
recon_ssim = ssim(image_rec, image)
#print('The Reconstruction SSIM is : ' + str(recon_ssim))
print(f'The Reconstruction SSIM is: {recon_ssim: 2f}')
