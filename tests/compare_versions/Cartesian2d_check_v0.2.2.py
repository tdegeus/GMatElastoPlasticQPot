import unittest

import GMatElastoPlasticQPot.Cartesian2d as GMat
import h5py
import numpy as np


class Test(unittest.TestCase):
    def test_main(self):

        with h5py.File("Cartesian2d_random.hdf5", "r") as data:

            shape = data["/shape"][...]

            i = np.eye(2)
            I = np.einsum("xy,ij", np.ones(shape), i)
            I4 = np.einsum("xy,ijkl->xyijkl", np.ones(shape), np.einsum("il,jk", i, i))
            I4rt = np.einsum("xy,ijkl->xyijkl", np.ones(shape), np.einsum("ik,jl", i, i))
            I4s = (I4 + I4rt) / 2.0

            mat = GMat.Matrix(shape[0], shape[1])

            isplastic = (data["/cusp/I"][...] + data["/smooth/I"][...]) > 0

            I = data["/cusp/I"][...]
            idx = data["/cusp/idx"][...]
            K = data["/cusp/K"][...]
            G = data["/cusp/G"][...]
            epsy = data["/cusp/epsy"][...][:, 1:]

            mat.setCusp(I, idx, K, G, epsy)

            I = data["/smooth/I"][...]
            idx = data["/smooth/idx"][...]
            K = data["/smooth/K"][...]
            G = data["/smooth/G"][...]
            epsy = data["/smooth/epsy"][...][:, 1:]

            mat.setSmooth(I, idx, K, G, epsy)

            I = data["/elastic/I"][...]
            idx = data["/elastic/idx"][...]
            K = data["/elastic/K"][...]
            G = data["/elastic/G"][...]

            mat.setElastic(I, idx, K, G)

            for i in range(20):

                GradU = data[f"/random/{i:d}/GradU"][...]

                Eps = np.einsum("...ijkl,...lk->...ij", I4s, GradU)
                idx = mat.Find(Eps)

                self.assertTrue(np.allclose(mat.Stress(Eps), data[f"/random/{i:d}/Stress"][...]))
                self.assertTrue(
                    np.allclose(
                        mat.Epsy(idx)[isplastic],
                        data[f"/random/{i:d}/CurrentYieldLeft"][...][isplastic],
                    )
                )
                self.assertTrue(
                    np.allclose(
                        mat.Epsy(idx + 1)[isplastic],
                        data[f"/random/{i:d}/CurrentYieldRight"][...][isplastic],
                    )
                )
                self.assertTrue(
                    np.all(
                        mat.Find(Eps)[isplastic]
                        == data[f"/random/{i:d}/CurrentIndex"][...][isplastic]
                    )
                )


if __name__ == "__main__":

    unittest.main()
