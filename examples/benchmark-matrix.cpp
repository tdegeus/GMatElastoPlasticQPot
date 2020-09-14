#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GMat = GMatElastoPlasticQPot::Cartesian2d;

int main()
{
    double K = 12.3;
    double G = 45.6;
    xt::xtensor<double,1> epsy = xt::linspace<double>(1e-5, 1, 5001);

    size_t nelem = 700;
    size_t nip = 9;

    GMat::Matrix mat(nelem, nip);
    xt::xtensor<double,4> Eps = xt::zeros<double>({nelem, nip, 2ul, 2ul});
    xt::xtensor<double,4> Sig = xt::zeros<double>({nelem, nip, 2ul, 2ul});

    xt::xtensor<size_t,2> I = xt::empty<size_t>({nelem, nip});

    I.fill(0);
    xt::view(I, xt::range(0, 500), xt::all()) = 1;
    mat.setElastic(I, K, G);

    I.fill(0);
    xt::view(I, xt::range(500, nelem), xt::all()) = 1;
    mat.setCusp(I, K, G, epsy);

    size_t n = 100000;
    for (size_t i = 0; i < n; ++i) {
        double g = 0.5 / static_cast<double>(n) * static_cast<double>(i);
        xt::view(Eps, xt::all(), xt::all(), 0, 1) = g;
        xt::view(Eps, xt::all(), xt::all(), 1, 0) = g;
        mat.stress(Eps, Sig);
    }


    return 0;
}
