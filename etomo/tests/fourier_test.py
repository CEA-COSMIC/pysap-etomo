import unittest
import numpy as np
from etomo.operators import NUFFT2, gpuNUFFT
from etomo.operators.utils import generate_locations_etomo_2D

try:
    import pynufft
except ImportError:
    import_pynufft = False
else:
    import_pynufft = True
try:
    import gpuNUFFT
except ImportError:
    use_gpu = False
else:
    use_gpu = True


class FourierTestCase(unittest.TestCase):
    """
    Computes <R.x,y> and <x,Rt.y> for random x and y for each operator and
    checks if the results are close enough.
    """
    def setUp(self) -> None:
        """
        Initialize test
        img_size: image size (square)
        nb_proj: number of projections
        """
        self.img_size = 200
        nb_proj = 100
        theta = np.arange(0.0, 180.0, 180.0 / nb_proj)
        self.kspace_loc = generate_locations_etomo_2D(self.img_size, theta)
        self.fake_data = np.random.rand(self.img_size, self.img_size)
        self.fake_adjoint_data = np.random.rand(
            nb_proj * int(np.sqrt(2) * self.img_size)
        )

    @unittest.skipUnless(import_pynufft, 'PyNUFFT not installed.')
    def test2D_cpu(self):
        """
        Tests adjoint operator of 2D operator
        """
        fourier = NUFFT2(2 * np.pi * self.kspace_loc, (self.img_size,) * 2)
        Fxy = np.sum(
            fourier.op(self.fake_data) * np.conj(self.fake_adjoint_data)
        )
        xFty = np.sum(
            self.fake_data * np.conj(fourier.adj_op(self.fake_adjoint_data))
        )
        self.assertTrue(np.allclose(Fxy, xFty, rtol=1e-6))

    @unittest.skipUnless(use_gpu, 'GPU not available.')
    def test2D_gpu(self):
        """
        Tests adjoint operator of 2D operator
        """
        fourier = gpuNUFFT(self.kspace_loc, (self.img_size,) * 2)
        Fxy = np.sum(
            fourier.op(self.fake_data) * np.conj(self.fake_adjoint_data)
        )
        xFty = np.sum(
            self.fake_data * np.conj(fourier.adj_op(self.fake_adjoint_data))
        )
        self.assertTrue(np.allclose(Fxy, xFty, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
