"""
Reconstruction using gpuNUFFT with density compensation
"""
# Third party import
import pysap
from pysap.data import get_sample_data
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np
import matplotlib.pyplot as plt

from etomo.operators.utils import generate_locations_etomo_2D
from etomo.operators.fourier.utils import estimate_density_compensation
from etomo.operators import gpuNUFFT, WaveletPywt, HOTV
from etomo.reconstructors.forwardradon import RadonReconstructor


# Loading input data
image = get_sample_data('2d-mri')
img_size = image.shape[0]

# Create radon operator and simulate data
theta = np.arange(0., 180., 3.)
kspace_loc = generate_locations_etomo_2D(img_size, theta)
density_comp = estimate_density_compensation(kspace_loc, image.shape)

fourier_op = gpuNUFFT(samples=kspace_loc, shape=image.shape,
                      density_comp=density_comp)
data = fourier_op.op(image.data)

TV = HOTV(img_shape=image.shape, order=1)
wavelet = WaveletPywt(wavelet_name='sym8', nb_scale=3)
linear_op = TV

regularizer_op = SparseThreshold(linear=Identity(), weights=5e-9)
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
    num_iterations=300,
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
