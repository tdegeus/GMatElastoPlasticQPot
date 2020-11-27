import unittest
import numpy as np
import GMatElastoPlasticQPot.Cartesian2d as GMat

class Test_main(unittest.TestCase):

    def test_Elastic(self):

        K = 12.3
        G = 45.6

        gamma = 0.02
        epsm = 0.12

        Eps = np.array(
            [[epsm, gamma],
             [gamma, epsm]])

        Sig = np.array(
            [[K * epsm, G * gamma],
             [G * gamma, K * epsm]])

        self.assertTrue(np.isclose(float(GMat.Epsd(Eps)), gamma))

        mat = GMat.Elastic(K, G)
        mat.setStrain(Eps)

        self.assertTrue(np.allclose(mat.Stress(), Sig))

    def test_Cusp(self):

        K = 12.3
        G = 45.6

        gamma = 0.02
        epsm = 0.12

        Eps = np.array(
            [[epsm, gamma],
             [gamma, epsm]])

        Sig = np.array(
            [[K * epsm, 0.0],
             [0.0, K * epsm]])

        self.assertTrue(np.isclose(float(GMat.Epsd(Eps)), gamma))

        mat = GMat.Cusp(K, G, [0.01, 0.03, 0.10])
        mat.setStrain(Eps)

        self.assertTrue(np.allclose(mat.Stress(), Sig))
        self.assertTrue(np.isclose(mat.epsp(), 0.02))
        self.assertTrue(mat.currentIndex() == 1)

    def test_Smooth(self):

        K = 12.3
        G = 45.6

        gamma = 0.02
        epsm = 0.12

        Eps = np.array(
            [[epsm, gamma],
             [gamma, epsm]])

        Sig = np.array(
            [[K * epsm, 0.0],
             [0.0, K * epsm]])

        self.assertTrue(np.isclose(float(GMat.Epsd(Eps)), gamma))

        mat = GMat.Smooth(K, G, [0.01, 0.03, 0.10])
        mat.setStrain(Eps)

        self.assertTrue(np.allclose(mat.Stress(), Sig))
        self.assertTrue(np.isclose(mat.epsp(), 0.02))
        self.assertTrue(mat.currentIndex() == 1)

    def test_Array2d(self):

        K = 12.3
        G = 45.6

        gamma = 0.02
        epsm = 0.12

        Eps = np.array(
            [[epsm, gamma],
             [gamma, epsm]])

        Sig_elas = np.array(
            [[K * epsm, G * gamma],
             [G * gamma, K * epsm]])

        Sig_plas = np.array(
            [[K * epsm, 0.0],
             [0.0, K * epsm]])

        nelem = 3
        nip = 2
        mat = GMat.Array2d([nelem, nip])

        I = np.zeros([nelem, nip], dtype='int')
        I[0,:] = 1
        mat.setElastic(I, K, G)

        I = np.zeros([nelem, nip], dtype='int')
        I[1,:] = 1
        mat.setCusp(I, K, G, 0.01 + 0.02 * np.arange(100))

        I = np.zeros([nelem, nip], dtype='int')
        I[2,:] = 1
        mat.setSmooth(I, K, G, 0.01 + 0.02 * np.arange(100))

        eps = np.zeros((nelem, nip, 2, 2))
        sig = np.zeros((nelem, nip, 2, 2))
        epsp = np.zeros((nelem, nip))

        for e in range(nelem):
            for q in range(nip):
                fac = float((e + 1) * nip + (q + 1))
                eps[e, q, :, :] = fac * Eps
                if e == 0:
                    sig[e, q, :, :] = fac * Sig_elas
                    epsp[e, q] = 0.0
                else:
                    sig[e, q, :, :] = fac * Sig_plas
                    epsp[e, q] = fac * gamma

        mat.setStrain(eps)

        self.assertTrue(np.allclose(mat.Stress(), sig))
        self.assertTrue(np.allclose(mat.Epsp(), epsp))

if __name__ == '__main__':

    unittest.main()
