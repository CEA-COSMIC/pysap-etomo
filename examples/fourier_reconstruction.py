"""
Example of a single image reconstruction with fourier operators. See
density_compensation for similar reconstruction with better results.

Credit: G. Biagi
"""
import numpy as np
import matplotlib.pyplot as plt

import pysap
from pysap.data import get_sample_data
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold

from etomo.operators.utils import generate_locations_etomo_2D
from etomo.operators import NUFFT2, WaveletPywt, HOTV
from etomo.reconstructors.forwardtomo import TomoReconstructor
from etomo.operators.fourier.utils import estimate_density_compensation


# %%
# Loading input data
image = get_sample_data('2d-mri')
img_size = image.shape[0]

# %%
# Create fourier operator and simulate data
theta = np.arange(0., 180., 3.)
kspace_loc = generate_locations_etomo_2D(img_size, theta)
fourier_op = NUFFT2(samples=kspace_loc, shape=image.shape, normalized=True)
data = fourier_op.op(image.data)

# %%
# Create operators
linear_op = WaveletPywt(wavelet_name='bior4.4', nb_scale=3)

regularizer_op = SparseThreshold(linear=Identity(), weights=1e-5)

# %%
# Run reconstruction
reconstructor = TomoReconstructor(
    data_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='analysis',
    verbose=True
)

x_final, cost, *_ = reconstructor.reconstruct(
    data=data,
    optimization_alg='condatvu',
    num_iterations=300,
    cost_op_kwargs={'cost_interval': 5}
)

# %%
# Results
fig, ax = plt.subplots(2, 2, figsize=(14, 14))

image_rec = pysap.Image(data=x_final)
ax[0][1].imshow(image_rec, cmap='gray')
ax[0][1].set_title('Reconstructed image')

ax[0][0].imshow(image, cmap='gray')
ax[0][0].set_title('Original image')

ax[1][0].plot(cost)
ax[1][0].set_yscale('log')
ax[1][0].set_title('Evolution of cost function')

ax[1][1].set_visible(False)

plt.show()

recon_ssim = ssim(image_rec, image)
print(f'The Reconstruction SSIM is: {recon_ssim: 2f}')
