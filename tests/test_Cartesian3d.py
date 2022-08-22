import unittest

import GMatElastic.Cartesian3d as GMatElastic
import GMatElastoPlasticQPot.Cartesian2d as GMat2d
import GMatElastoPlasticQPot.Cartesian3d as GMat
import GMatTensor.Cartesian3d as tensor
import numpy as np


class Test_main(unittest.TestCase):
    """ """

    def test_Epsd_Sigd(self):

        A = np.zeros((2, 3, 3, 3))
        A[..., 0, 1] = 1
        A[..., 1, 0] = 1

        self.assertTrue(np.allclose(GMat.Epsd(A), np.ones(A.shape[:-2])))
        self.assertTrue(np.allclose(GMat.Sigd(A), 2 * np.ones(A.shape[:-2])))

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
        mat.Eps[..., 2, 2] = epsm
        mat.Eps[..., 0, 1] = gamma
        mat.Eps[..., 1, 0] = gamma
        mat.refresh()
        smooth.Eps = mat.Eps

        Sig = np.zeros(shape + [3, 3])
        Sig[..., 0, 0] = 3 * K * epsm
        Sig[..., 1, 1] = 3 * K * epsm
        Sig[..., 2, 2] = 3 * K * epsm
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
        self.assertTrue(np.allclose(mat.energy, 3 * K * epsm**2 - 2 * G * 0.01**2))

        mat.epsy = np.zeros(shape + [3])
        mat.epsy += np.array([0.01, 0.03, 0.10]).reshape([1, 1, -1])

        self.assertTrue(np.allclose(GMat.Epsd(mat.Eps), gamma))
        self.assertTrue(np.allclose(GMat.Sigd(mat.Sig), 0))
        self.assertTrue(np.allclose(mat.Sig, Sig))
        self.assertTrue(np.all(mat.i == 0))
        self.assertTrue(np.allclose(mat.epsp, 0.02))
        self.assertTrue(np.allclose(mat.epsy_left, 0.01))
        self.assertTrue(np.allclose(mat.epsy_right, 0.03))
        self.assertTrue(np.allclose(mat.energy, 3 * K * epsm**2 - 2 * G * 0.01**2))

        mat.epsy = np.zeros(shape + [5])
        mat.epsy += np.array([-0.01, 0.01, 0.03, 0.05, 0.10]).reshape([1, 1, -1])

        gamma = 0.04 + 0.009 * np.random.random(shape)
        mat.Eps[..., 0, 1] = gamma
        mat.Eps[..., 1, 0] = gamma
        mat.refresh()

        Sig[..., 0, 1] = 2 * G * (gamma - 0.04)
        Sig[..., 1, 0] = 2 * G * (gamma - 0.04)

        self.assertTrue(np.allclose(GMat.Epsd(mat.Eps), gamma))
        self.assertTrue(np.allclose(GMat.Sigd(mat.Sig), 4 * G * np.abs(gamma - 0.04)))
        self.assertTrue(np.allclose(mat.Sig, Sig))
        self.assertTrue(np.all(mat.i == 2))
        self.assertTrue(np.allclose(mat.epsp, 0.04))
        self.assertTrue(np.allclose(mat.epsy_left, 0.03))
        self.assertTrue(np.allclose(mat.epsy_right, 0.05))
        self.assertTrue(
            np.allclose(mat.energy, 3 * K * epsm**2 + 2 * G * ((gamma - 0.04) ** 2 - 0.01**2))
        )

        mat.Eps *= 0
        smooth.Eps *= 0

        self.assertTrue(np.allclose(mat.Sig, 0 * Sig))
        self.assertTrue(np.allclose(smooth.Sig, 0 * Sig))
        self.assertTrue(np.allclose(mat.energy, -2 * G * 0.01**2))
        self.assertTrue(np.allclose(smooth.energy, -4 * G * (0.01 / np.pi) ** 2 * 2))

    def test_Elastic_2d(self):

        shape = [2, 3]
        K = np.random.random(shape)
        G = np.random.random(shape)

        Eps = tensor.A4_ddot_B2(tensor.Array2d(shape).I4s, np.random.random(shape + [3, 3]))
        Eps[..., :, 2] = 0
        Eps[..., 2, :] = 0
        Eps[..., 0, 0] = -Eps[..., 1, 1]

        mat2d = GMat2d.Elastic2d(3 * K, 2 * G)
        mat3d = GMatElastic.Elastic2d(K, G)

        mat2d.Eps = Eps[..., :2, :2]
        mat3d.Eps = Eps

        self.assertTrue(np.allclose(mat2d.Sig, mat3d.Sig[..., :2, :2]))
        self.assertTrue(np.allclose(mat2d.energy, mat3d.energy))

    def test_Cusp_2d(self):

        shape = [2, 3]
        K = np.random.random(shape)
        G = np.random.random(shape)
        epsy = np.zeros(shape + [5])
        epsy += np.array([-0.01, 0.01, 0.5, 1, 2]).reshape([1, 1, -1])

        Eps = tensor.A4_ddot_B2(tensor.Array2d(shape).I4s, np.random.random(shape + [3, 3]))
        Eps[..., :, 2] = 0
        Eps[..., 2, :] = 0
        Eps[..., 0, 0] = -Eps[..., 1, 1]

        mat2d = GMat2d.Cusp2d(3 * K, 2 * G, epsy)
        mat3d = GMat.Cusp2d(K, G, epsy)

        mat2d.Eps = Eps[..., :2, :2]
        mat3d.Eps = Eps

        self.assertTrue(np.allclose(mat2d.Sig, mat3d.Sig[..., :2, :2]))
        self.assertTrue(np.allclose(mat2d.energy, mat3d.energy))
        self.assertTrue(np.allclose(mat2d.epsp, mat3d.epsp))
        self.assertTrue(np.allclose(mat2d.epsy_left, mat3d.epsy_left))
        self.assertTrue(np.allclose(mat2d.epsy_right, mat3d.epsy_right))

    def test_Smooth_2d(self):

        shape = [2, 3]
        K = np.random.random(shape)
        G = np.random.random(shape)
        epsy = np.zeros(shape + [5])
        epsy += np.array([-0.01, 0.01, 0.5, 1, 2]).reshape([1, 1, -1])

        Eps = tensor.A4_ddot_B2(tensor.Array2d(shape).I4s, np.random.random(shape + [3, 3]))
        Eps[..., :, 2] = 0
        Eps[..., 2, :] = 0
        Eps[..., 0, 0] = -Eps[..., 1, 1]

        mat2d = GMat2d.Smooth2d(3 * K, 2 * G, epsy)
        mat3d = GMat.Smooth2d(K, G, epsy)

        mat2d.Eps = Eps[..., :2, :2]
        mat3d.Eps = Eps

        self.assertTrue(np.allclose(mat2d.Sig, mat3d.Sig[..., :2, :2]))
        self.assertTrue(np.allclose(mat2d.energy, mat3d.energy))
        self.assertTrue(np.allclose(mat2d.epsp, mat3d.epsp))
        self.assertTrue(np.allclose(mat2d.epsy_left, mat3d.epsy_left))
        self.assertTrue(np.allclose(mat2d.epsy_right, mat3d.epsy_right))


if __name__ == "__main__":

    unittest.main()
