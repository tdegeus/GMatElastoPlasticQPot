import unittest

import GMatElastoPlasticQPot.Cartesian2d as GMat
import h5py
import numpy as np


class Test(unittest.TestCase):
    def test_main(self):

        with h5py.File("Cartesian2d_random.hdf5", "r") as data:

            mat = GMat.Array2d(data["/shape"][...])

            I = data["/cusp/I"][...]
            idx = data["/cusp/idx"][...]
            K = data["/cusp/K"][...]
            G = data["/cusp/G"][...]
            epsy = data["/cusp/epsy"][...]

            mat.setCusp(I, idx, K, G, epsy)

            I = data["/smooth/I"][...]
            idx = data["/smooth/idx"][...]
            K = data["/smooth/K"][...]
            G = data["/smooth/G"][...]
            epsy = data["/smooth/epsy"][...]

            mat.setSmooth(I, idx, K, G, epsy)

            I = data["/elastic/I"][...]
            idx = data["/elastic/idx"][...]
            K = data["/elastic/K"][...]
            G = data["/elastic/G"][...]

            mat.setElastic(I, idx, K, G)

            for i in range(20):

                GradU = data[f"/random/{i:d}/GradU"][...]

                Eps = np.einsum("...ijkl,...lk->...ij", mat.I4s(), GradU)
                mat.setStrain(Eps)

                self.assertTrue(np.allclose(mat.Stress(), data[f"/random/{i:d}/Stress"][...]))
                self.assertTrue(np.allclose(mat.Tangent(), data[f"/random/{i:d}/Tangent"][...]))
                self.assertTrue(
                    np.allclose(
                        mat.CurrentYieldLeft(),
                        data[f"/random/{i:d}/CurrentYieldLeft"][...],
                    )
                )
                self.assertTrue(
                    np.allclose(
                        mat.CurrentYieldRight(),
                        data[f"/random/{i:d}/CurrentYieldRight"][...],
                    )
                )
                self.assertTrue(
                    np.all(mat.CurrentIndex() == data[f"/random/{i:d}/CurrentIndex"][...])
                )


if __name__ == "__main__":

    unittest.main()
