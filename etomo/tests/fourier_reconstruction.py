"""
Example of a single image reconstruction
"""
import numpy as np
import matplotlib.pyplot as plt

import pysap
from pysap.data import get_sample_data
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold

from etomo.operators.utils import generate_locations_etomo_2D
from etomo.operators import NFFT, WaveletPywt, HOTV
from etomo.reconstructors.forwardradon import RadonReconstructor
from etomo.operators.fourier.utils import estimate_density_compensation


# Loading input data
image = get_sample_data('2d-mri')
img_size = image.shape[0]

# Create radon operator and simulate data
theta = np.arange(0., 180., 3.)
kspace_loc = generate_locations_etomo_2D(img_size, theta)
fourier_op = NFFT(samples=kspace_loc, shape=image.shape, normalized=True)
data = fourier_op.op(image.data)

TV = HOTV(img_shape=image.shape, order=1)
wavelet = WaveletPywt(wavelet_name='sym8', nb_scale=3)
linear_op = TV

regularizer_op = SparseThreshold(linear=Identity(), weights=3e-6)
reconstructor = RadonReconstructor(
    data_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='analysis',
    verbose=True
)

# Run reconstruction
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

image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
recon_ssim = ssim(image_rec, image)
print(f'The Reconstruction SSIM is {recon_ssim : 2f} ')
