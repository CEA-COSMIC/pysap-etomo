import unittest
import numpy as np
from pysap.data import get_sample_data

from etomo.operators import NUFFT2, NFFT, gpuNUFFT
from etomo.operators.utils import generate_locations_etomo_2D


class MyTestCase(unittest.TestCase):
    """
    Computes <R.x,y> and <x,Rt.y> for random x and y for each operator and checks
    if the results are close enough.
    """
    def setUp(self) -> None:
        """
        Initialize test
        img_size: image size (square)
        nb_proj: number of projections
        """
        self.img_size = 200
        self.nb_proj = 100

    def test2D(self):
        """
        Tests adjoint operator of 2D operator
        """
        theta = np.arange(0., 180., 180./self.nb_proj)
        kspace_loc = generate_locations_etomo_2D(self.img_size, theta)

        fourier_pynufft = NUFFT2(2 * np.pi * kspace_loc, (self.img_size,) * 2)
        fourier_pynfft = NFFT(kspace_loc, (self.img_size,) * 2)
        fourier_gpu = gpuNUFFT(kspace_loc, (self.img_size,) * 2)

        fake_data = np.random.rand(self.img_size, self.img_size)
        fake_adjoint_data = np.random.rand(self.nb_proj * int(
            np.sqrt(2) * self.img_size))

        for ope in [fourier_pynfft, fourier_pynufft, fourier_gpu]:
            Fxy = np.sum(ope.op(fake_data) * np.conj(fake_adjoint_data))
            xFty = np.sum(fake_data * np.conj(ope.adj_op(fake_adjoint_data)))
            print(Fxy, xFty)
            self.assertTrue(np.allclose(Fxy, xFty, rtol=1e-6))

    def test_results(self):
        """
        Compares results from pynfft and pynufft
        """
        image = get_sample_data('2d-mri').data
        theta = np.arange(0., 180., 180. / self.nb_proj)
        kspace_loc = generate_locations_etomo_2D(image.shape[0], theta)

        fourier_pynufft = NUFFT2(2 * np.pi * kspace_loc, image.shape)
        fourier_pynfft = NFFT(kspace_loc, image.shape)

        pynfft_data = fourier_pynfft.op(image)
        pynufft_data = fourier_pynufft.op(image)
        print(pynufft_data)
        print(pynfft_data)
        self.assertTrue(np.allclose(pynufft_data, pynfft_data, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
