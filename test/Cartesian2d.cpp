
#include <catch2/catch.hpp>

#define EQ(a,b) REQUIRE_THAT((a), Catch::WithinAbs((b), 1.e-12));

#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GM = GMatElastoPlasticQPot::Cartesian2d;

TEST_CASE("GMatElastoPlasticQPot::Cartesian2d", "Cartesian2d.h")
{

double K = 12.3;
double G = 45.6;

GM::Tensor2 Eps;
GM::Tensor2 Sig;

double gamma = 0.02;
double epsm = 0.12;

Eps(0,0) = Eps(1,1) = epsm;
Eps(0,1) = Eps(1,0) = gamma;

SECTION("Elastic: simple shear + volumetric deformation")
{
    GM::Elastic mat(K, G);

    Sig = mat.Stress(Eps);

    EQ(Sig(0,0), K * epsm);
    EQ(Sig(1,1), K * epsm);
    EQ(Sig(0,1), G * gamma);
    EQ(Sig(1,0), G * gamma);
}

SECTION("Cusp: simple shear + volumetric deformation")
{
    GM::Cusp mat(K, G, {0.01, 0.03, 0.10});

    Sig = mat.Stress(Eps);

    EQ(Sig(0,0), K * epsm);
    EQ(Sig(1,1), K * epsm);
    EQ(Sig(0,1), 0.0);
    EQ(Sig(1,0), 0.0);

    EQ(mat.epsp(Eps), 0.02);

    REQUIRE(mat.find(Eps) == 1);
}

SECTION("Smooth: simple shear + volumetric deformation")
{
    GM::Smooth mat(K, G, {0.01, 0.03, 0.10});

    Sig = mat.Stress(Eps);

    EQ(Sig(0,0), K * epsm);
    EQ(Sig(1,1), K * epsm);
    EQ(Sig(0,1), 0.0);
    EQ(Sig(1,0), 0.0);

    EQ(mat.epsp(Eps), 0.02);

    REQUIRE(mat.find(Eps) == 1);
}

SECTION("Matrix")
{
    size_t nelem = 3;
    size_t nip = 2;

    GM::Matrix mat(nelem, nip);

    // row 0: elastic
    {
      xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
      xt::view(I, 0, xt::all()) = 1;
      mat.setElastic(I, K, G);
    }

    // row 1: cups
    {
      xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
      xt::xtensor<double,1> epsy = {0.01, 0.03, 0.10};
      xt::view(I, 1, xt::all()) = 1;
      mat.setCusp(I, K, G, epsy);
    }

    // row 2: smooth
    {
      xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
      xt::xtensor<double,1> epsy = {0.01, 0.03, 0.10};
      xt::view(I, 2, xt::all()) = 1;
      mat.setCusp(I, K, G, epsy);
    }

    xt::xtensor<double,4> eps = xt::empty<double>({nelem, nip, 2ul, 2ul});

    for (size_t e = 0; e < nelem; ++e) {
        for (size_t q = 0; q < nip; ++q) {
            xt::view(eps, e, q) = Eps;
        }
    }

    auto sig = mat.Stress(eps);
    auto epsp = mat.Epsp(eps);

    for (size_t q = 0; q < nip; ++q) {
        EQ(sig(0,q,0,0), K * epsm);
        EQ(sig(0,q,1,1), K * epsm);
        EQ(sig(0,q,0,1), G * gamma);
        EQ(sig(0,q,1,0), G * gamma);
        EQ(sig(1,q,0,0), K * epsm);
        EQ(sig(1,q,1,1), K * epsm);
        EQ(sig(1,q,0,1), 0.0);
        EQ(sig(1,q,1,0), 0.0);
        EQ(sig(2,q,0,0), K * epsm);
        EQ(sig(2,q,1,1), K * epsm);
        EQ(sig(2,q,0,1), 0.0);
        EQ(sig(2,q,1,0), 0.0);

        EQ(epsp(0,q), 0.0);
        EQ(epsp(1,q), gamma);
        EQ(epsp(2,q), gamma);
    }
}

}
