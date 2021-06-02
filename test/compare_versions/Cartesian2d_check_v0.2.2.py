import h5py
import numpy as np
import GMatElastoPlasticQPot.Cartesian2d as GMat
import unittest

class Test(unittest.TestCase):

    def test_main(self):

        with h5py.File('Cartesian2d_random.hdf5', 'r') as data:

            shape = data['/shape'][...]

            i = np.eye(2)
            I = np.einsum('xy,ij', np.ones(shape), i)
            I4 = np.einsum('xy,ijkl->xyijkl', np.ones(shape), np.einsum('il,jk', i, i))
            I4rt = np.einsum('xy,ijkl->xyijkl', np.ones(shape), np.einsum('ik,jl', i, i))
            I4s = (I4 + I4rt) / 2.0

            mat = GMat.Matrix(shape[0], shape[1])

            I = data['/cusp/I'][...]
            idx = data['/cusp/idx'][...]
            K = data['/cusp/K'][...]
            G = data['/cusp/G'][...]
            epsy = data['/cusp/epsy'][...]

            mat.setCusp(I, idx, K, G, epsy)

            I = data['/smooth/I'][...]
            idx = data['/smooth/idx'][...]
            K = data['/smooth/K'][...]
            G = data['/smooth/G'][...]
            epsy = data['/smooth/epsy'][...]

            mat.setSmooth(I, idx, K, G, epsy)

            I = data['/elastic/I'][...]
            idx = data['/elastic/idx'][...]
            K = data['/elastic/K'][...]
            G = data['/elastic/G'][...]

            mat.setElastic(I, idx, K, G)

            for i in range(20):

                GradU = data['/random/{0:d}/GradU'.format(i)][...]

                Eps = np.einsum('...ijkl,...lk->...ij', I4s, GradU)
                idx = mat.Find(Eps)

                self.assertTrue(np.allclose(mat.Stress(Eps), data['/random/{0:d}/Stress'.format(i)][...]))
                self.assertTrue(np.allclose(mat.Epsy(idx), data['/random/{0:d}/CurrentYieldLeft'.format(i)][...]))
                self.assertTrue(np.allclose(mat.Epsy(idx + 1), data['/random/{0:d}/CurrentYieldRight'.format(i)][...]))
                self.assertTrue(np.all(mat.Find(Eps) == data['/random/{0:d}/CurrentIndex'.format(i)][...] - 1))

if __name__ == '__main__':

    unittest.main()
