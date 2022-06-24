import unittest

import GMatElastoPlasticQPot.Cartesian2d as GMat
import GMatTensor.Cartesian2d as tensor
import numpy as np


class Test_main(unittest.TestCase):
    """ """

    def test_Epsd_Sigd(self):

        A = np.zeros((2, 3, 2, 2))
        A[..., 0, 1] = 1
        A[..., 1, 0] = 1

        self.assertTrue(np.allclose(GMat.Epsd(A), np.ones(A.shape[:-2])))
        self.assertTrue(np.allclose(GMat.Sigd(A), 2 * np.ones(A.shape[:-2])))

    def test_Elastic(self):

        shape = [2, 3]
        K = np.random.random(shape)
        G = np.random.random(shape)
        mat = GMat.Elastic2d(K, G)

        gamma = np.random.random(shape)
        epsm = np.random.random(shape)

        mat.Eps[..., 0, 0] = epsm
        mat.Eps[..., 1, 1] = epsm
        mat.Eps[..., 0, 1] = gamma
        mat.Eps[..., 1, 0] = gamma
        mat.refresh()

        Sig = np.empty(shape + [2, 2])
        Sig[..., 0, 0] = K * epsm
        Sig[..., 1, 1] = K * epsm
        Sig[..., 0, 1] = G * gamma
        Sig[..., 1, 0] = G * gamma

        self.assertTrue(np.allclose(GMat.Epsd(mat.Eps), gamma))
        self.assertTrue(np.allclose(GMat.Sigd(mat.Sig), 2 * G * gamma))
        self.assertTrue(np.allclose(mat.Sig, Sig))
        self.assertTrue(np.allclose(tensor.A4_ddot_B2(mat.C, mat.Eps), Sig))
        self.assertTrue(np.allclose(mat.energy, K * epsm**2 + G * gamma**2))
        self.assertTrue(np.allclose(mat.K, K))
        self.assertTrue(np.allclose(mat.G, G))

    def test_tangent(self):

        shape = [2, 3]
        Eps = np.random.random(shape + [2, 2])
        Eps = tensor.A4_ddot_B2(tensor.Array2d(shape).I4s, Eps)
        mat = GMat.Elastic2d(np.random.random(shape), np.random.random(shape))
        mat.Eps = Eps
        self.assertTrue(np.allclose(tensor.A4_ddot_B2(mat.C, mat.Eps), mat.Sig))

    def test_Cusp_Smooth(self):

        shape = [2, 3]
        K = np.random.random(shape)
        G = np.random.random(shape)
        epsy = np.zeros(shape + [4])
        epsy += np.array([-0.01, 0.01, 0.03, 0.10]).reshape([1, 1, -1])
        mat = GMat.Cusp2d(K, G, epsy)
        smooth = GMat.Smooth2d(K, G, epsy)

        gamma = 0.02
        epsm = np.random.random(shape)

        mat.Eps[..., 0, 0] = epsm
        mat.Eps[..., 1, 1] = epsm
        mat.Eps[..., 0, 1] = gamma
        mat.Eps[..., 1, 0] = gamma
        mat.refresh()
        smooth.Eps = mat.Eps

        Sig = np.empty(shape + [2, 2])
        Sig[..., 0, 0] = K * epsm
        Sig[..., 1, 1] = K * epsm
        Sig[..., 0, 1] = 0
        Sig[..., 1, 0] = 0

        self.assertTrue(np.allclose(GMat.Epsd(mat.Eps), gamma))
        self.assertTrue(np.allclose(GMat.Sigd(mat.Sig), 0))
        self.assertTrue(np.allclose(mat.Sig, Sig))
        self.assertTrue(np.allclose(smooth.Sig, Sig))
        self.assertTrue(np.all(mat.i == 1))
        self.assertTrue(np.allclose(mat.epsp, 0.02))
        self.assertTrue(np.allclose(mat.epsy_left, 0.01))
        self.assertTrue(np.allclose(mat.epsy_right, 0.03))
        self.assertTrue(np.allclose(mat.energy, K * epsm**2 - G * 0.01**2))

        mat.epsy = np.zeros(shape + [3])
        mat.epsy += np.array([0.01, 0.03, 0.10]).reshape([1, 1, -1])

        self.assertTrue(np.allclose(GMat.Epsd(mat.Eps), gamma))
        self.assertTrue(np.allclose(GMat.Sigd(mat.Sig), 0))
        self.assertTrue(np.allclose(mat.Sig, Sig))
        self.assertTrue(np.all(mat.i == 0))
        self.assertTrue(np.allclose(mat.epsp, 0.02))
        self.assertTrue(np.allclose(mat.epsy_left, 0.01))
        self.assertTrue(np.allclose(mat.epsy_right, 0.03))
        self.assertTrue(np.allclose(mat.energy, K * epsm**2 - G * 0.01**2))

        mat.epsy = np.zeros(shape + [5])
        mat.epsy += np.array([-0.01, 0.01, 0.03, 0.05, 0.10]).reshape([1, 1, -1])

        gamma = 0.04 + 0.009 * np.random.random(shape)
        mat.Eps[..., 0, 1] = gamma
        mat.Eps[..., 1, 0] = gamma
        mat.refresh()

        Sig[..., 0, 1] = G * (gamma - 0.04)
        Sig[..., 1, 0] = G * (gamma - 0.04)

        self.assertTrue(np.allclose(GMat.Epsd(mat.Eps), gamma))
        self.assertTrue(np.allclose(GMat.Sigd(mat.Sig), 2 * G * np.abs(gamma - 0.04)))
        self.assertTrue(np.allclose(mat.Sig, Sig))
        self.assertTrue(np.all(mat.i == 2))
        self.assertTrue(np.allclose(mat.epsp, 0.04))
        self.assertTrue(np.allclose(mat.epsy_left, 0.03))
        self.assertTrue(np.allclose(mat.epsy_right, 0.05))
        self.assertTrue(
            np.allclose(mat.energy, K * epsm**2 + G * ((gamma - 0.04) ** 2 - 0.01**2))
        )

        mat.Eps *= 0
        smooth.Eps *= 0

        self.assertTrue(np.allclose(mat.Sig, 0 * Sig))
        self.assertTrue(np.allclose(smooth.Sig, 0 * Sig))
        self.assertTrue(np.allclose(mat.energy, -G * 0.01**2))
        self.assertTrue(np.allclose(smooth.energy, -2 * G * (0.01 / np.pi) ** 2 * 2))


if __name__ == "__main__":

    unittest.main()
