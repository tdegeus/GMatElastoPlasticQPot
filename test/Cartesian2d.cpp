
#include <catch2/catch.hpp>

#define EQ(a,b) REQUIRE_THAT( (a), Catch::WithinAbs((b), 1.e-12) );

#include "../include/GMatElastoPlasticQPot/Cartesian2d.h"

namespace GM = GMatElastoPlasticQPot::Cartesian2d;

// =================================================================================================

TEST_CASE("GMatElastoPlasticQPot::Cartesian2d", "Cartesian2d.h")
{

// =================================================================================================

SECTION("Elastic")
{
  // material model
  // - parameters
  double K = 12.3;
  double G = 45.6;
  // - model
  GM::Elastic mat(K,G);

  // allocate tensors
  GM::T2 Eps, Sig;

  // simple shear + volumetric deformation
  // - parameters
  double gamma = 0.02;
  double epsm  = 0.12;
  // - strain
  Eps(0,0) = Eps(1,1) = epsm;
  Eps(0,1) = Eps(1,0) = gamma;
  // - stress
  Sig = mat.Stress(Eps);
  // - analytical solution
  EQ(Sig(0,0), K * epsm);
  EQ(Sig(1,1), K * epsm);
  EQ(Sig(0,1), G * gamma);
  EQ(Sig(1,0), G * gamma);
  // - plastic strain
  EQ(mat.epsp(Eps), 0);
  // - yield strain index
  REQUIRE(mat.find(Eps) == 0);
}

// =================================================================================================

SECTION("Cusp")
{
  // material model
  // - parameters
  double K = 12.3;
  double G = 45.6;
  // - model
  GM::Cusp mat(K,G,{0.01, 0.03, 0.10});

  // allocate tensors
  GM::T2 Eps, Sig;

  // simple shear + volumetric deformation
  // - parameters
  double gamma = 0.02;
  double epsm  = 0.12;
  // - strain
  Eps(0,0) = Eps(1,1) = epsm;
  Eps(0,1) = Eps(1,0) = gamma;
  // - stress
  Sig = mat.Stress(Eps);
  // - analytical solution
  EQ(Sig(0,0), K * epsm);
  EQ(Sig(1,1), K * epsm);
  EQ(Sig(0,1), 0.);
  EQ(Sig(1,0), 0.);
  // - plastic strain
  EQ(mat.epsp(Eps), 0.02);
  // - yield strain index
  REQUIRE(mat.find(Eps) == 1);
}

// =================================================================================================

SECTION("Smooth")
{
  // material model
  // - parameters
  double K = 12.3;
  double G = 45.6;
  // - model
  GM::Smooth mat(K,G,{0.01, 0.03, 0.10});

  // allocate tensors
  GM::T2 Eps, Sig;

  // simple shear + volumetric deformation
  // - parameters
  double gamma = 0.02;
  double epsm  = 0.12;
  // - strain
  Eps(0,0) = Eps(1,1) = epsm;
  Eps(0,1) = Eps(1,0) = gamma;
  // - stress
  Sig = mat.Stress(Eps);
  // - analytical solution
  EQ(Sig(0,0), K * epsm);
  EQ(Sig(1,1), K * epsm);
  EQ(Sig(0,1), 0.);
  EQ(Sig(1,0), 0.);
  // - plastic strain
  EQ(mat.epsp(Eps), 0.02);
  // - yield strain index
  REQUIRE(mat.find(Eps) == 1);
}

// =================================================================================================

SECTION("Matrix")
{
  // parameters
  double K     = 12.3;
  double G     = 45.6;
  size_t nelem = 3;
  size_t nip   = 2;

  // allocate matrix
  GM::Matrix mat({nelem,nip});

  // row 0: elastic
  {
    xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem,nip});
    xt::view(I, xt::keep(0), xt::all()) = 1;
    mat.setElastic(I,K,G);
  }

  // row 1: cups
  {
    xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem,nip});
    xt::xtensor<double,1> epsy = {0.01, 0.03, 0.10};
    xt::view(I, xt::keep(1), xt::all()) = 1;
    mat.setCusp(I,K,G,epsy);
  }

  // row 2: smooth
  {
    xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem,nip});
    xt::xtensor<double,1> epsy = {0.01, 0.03, 0.10};
    xt::view(I, xt::keep(2), xt::all()) = 1;
    mat.setCusp(I,K,G,epsy);
  }

  // allocate tensors
  GM::T2 Eps;

  // simple shear + volumetric deformation
  // - parameters
  double gamma = 0.02;
  double epsm  = 0.12;
  // - strain
  Eps(0,0) = Eps(1,1) = epsm;
  Eps(0,1) = Eps(1,0) = gamma;
  // - strain/stress matrices
  xt::xtensor<double,4> eps = xt::empty<double>({3,2,2,2});
  xt::xtensor<double,4> sig;
  xt::xtensor<double,2> epsp;
  // - set strain
  for (size_t e = 0; e < nelem; ++e)
    for (size_t q = 0; q < nip; ++q)
      xt::view(eps, e, q) = Eps;
  // - stress & plastic strain
  sig  = mat.Stress(eps);
  epsp = mat.Epsp(eps);

  // - analytical solution
  EQ(sig(0,0,0,0), K * epsm ); EQ(sig(0,1,0,0), K * epsm );
  EQ(sig(0,0,1,1), K * epsm ); EQ(sig(0,1,1,1), K * epsm );
  EQ(sig(0,0,0,1), G * gamma); EQ(sig(0,1,0,1), G * gamma);
  EQ(sig(0,0,1,0), G * gamma); EQ(sig(0,1,1,0), G * gamma);
  EQ(sig(1,0,0,0), K * epsm ); EQ(sig(1,1,0,0), K * epsm );
  EQ(sig(1,0,1,1), K * epsm ); EQ(sig(1,1,1,1), K * epsm );
  EQ(sig(1,0,0,1), 0.       ); EQ(sig(1,1,0,1), 0.       );
  EQ(sig(1,0,1,0), 0.       ); EQ(sig(1,1,1,0), 0.       );
  EQ(sig(2,0,0,0), K * epsm ); EQ(sig(2,1,0,0), K * epsm );
  EQ(sig(2,0,1,1), K * epsm ); EQ(sig(2,1,1,1), K * epsm );
  EQ(sig(2,0,0,1), 0.       ); EQ(sig(2,1,0,1), 0.       );
  EQ(sig(2,0,1,0), 0.       ); EQ(sig(2,1,1,0), 0.       );
  // - plastic strain
  EQ(epsp(0,0), 0    ); EQ(epsp(0,1), 0    );
  EQ(epsp(1,0), gamma); EQ(epsp(1,1), gamma);
  EQ(epsp(2,0), gamma); EQ(epsp(2,1), gamma);
}

// =================================================================================================

}
