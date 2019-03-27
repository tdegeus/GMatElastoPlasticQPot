/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

================================================================================================= */

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

// -------------------------------------------------------------------------------------------------

template<class T>
inline double trace(const T& A)
{
  return A(0,0) + A(1,1);
}

// -------------------------------------------------------------------------------------------------

template <class T>
inline double ddot22(const T& A, const T& B)
{
  return A(0,0) * B(0,0) + 2.0 * A(0,1) * B(0,1) + A(1,1) * B(1,1);
}

// -------------------------------------------------------------------------------------------------

inline T2 I()
{
  return T2({{1., 0.},
             {0., 1.}});
}

// -------------------------------------------------------------------------------------------------

inline double Hydrostatic(const T2& A)
{
  return 0.5 * trace(A);
}

// -------------------------------------------------------------------------------------------------

inline T2 Deviatoric(const T2& A)
{
  return A - 0.5 * trace(A) * I();
}

// -------------------------------------------------------------------------------------------------

inline double Epsd(const T2& Eps)
{
  T2 Epsd = Eps - 0.5 * trace(Eps) * I();
  return std::sqrt(0.5 * ddot22(Epsd,Epsd));
}

// -------------------------------------------------------------------------------------------------

inline double Sigd(const T2& Sig)
{
  T2 Sigd = Sig - 0.5 * trace(Sig) * I();
  return std::sqrt(2.0 * ddot22(Sigd,Sigd));
}

// -------------------------------------------------------------------------------------------------

inline void hydrostatic(const xt::xtensor<double,4>& A, xt::xtensor<double,2>& Am)
{
  GMATELASTOPLASTICQPOT_ASSERT(A.shape() ==\
   std::decay_t<decltype(A)>::shape_type({Am.shape()[0], Am.shape()[1], 2, 2}));

  #pragma omp parallel
  {
    #pragma omp for
    for (size_t e = 0; e < A.shape()[0]; ++e) {
      for (size_t q = 0; q < A.shape()[1]; ++q) {
        auto Ai = xt::adapt(&A(e,q,0,0), xt::xshape<2,2>());
        Am(e,q) = 0.5 * trace(Ai);
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void deviatoric(const xt::xtensor<double,4>& A, xt::xtensor<double,4>& Ad)
{
  GMATELASTOPLASTICQPOT_ASSERT(A.shape() == Ad.shape());
  GMATELASTOPLASTICQPOT_ASSERT(A.shape() ==\
   std::decay_t<decltype(A)>::shape_type({Ad.shape()[0], Ad.shape()[1], 2, 2}));

  #pragma omp parallel
  {
    T2 I = Cartesian2d::I();
    #pragma omp for
    for (size_t e = 0; e < A.shape()[0]; ++e) {
      for (size_t q = 0; q < A.shape()[1]; ++q) {
        auto Ai  = xt::adapt(&A (e,q,0,0), xt::xshape<2,2>());
        auto Aid = xt::adapt(&Ad(e,q,0,0), xt::xshape<2,2>());
        xt::noalias(Aid) = Ai - 0.5 * trace(Ai) * I;
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void epsd(const xt::xtensor<double,4>& A, xt::xtensor<double,2>& Aeq)
{
  GMATELASTOPLASTICQPOT_ASSERT(A.shape() ==\
   std::decay_t<decltype(A)>::shape_type({Aeq.shape()[0], Aeq.shape()[1], 2, 2}));

  #pragma omp parallel
  {
    T2 I = Cartesian2d::I();
    #pragma omp for
    for (size_t e = 0; e < A.shape()[0]; ++e) {
      for (size_t q = 0; q < A.shape()[1]; ++q) {
        auto Ai  = xt::adapt(&A(e,q,0,0), xt::xshape<2,2>());
        auto Aid = Ai - 0.5 * trace(Ai) * I;
        Aeq(e,q) = std::sqrt(0.5 * ddot22(Aid,Aid));
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void sigd(const xt::xtensor<double,4>& A, xt::xtensor<double,2>& Aeq)
{
  GMATELASTOPLASTICQPOT_ASSERT(A.shape() ==\
   std::decay_t<decltype(A)>::shape_type({Aeq.shape()[0], Aeq.shape()[1], 2, 2}));

  #pragma omp parallel
  {
    T2 I = Cartesian2d::I();
    #pragma omp for
    for (size_t e = 0; e < A.shape()[0]; ++e) {
      for (size_t q = 0; q < A.shape()[1]; ++q) {
        auto Ai  = xt::adapt(&A(e,q,0,0), xt::xshape<2,2>());
        auto Aid = Ai - 0.5 * trace(Ai) * I;
        Aeq(e,q) = std::sqrt(2.0 * ddot22(Aid,Aid));
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Hydrostatic(const xt::xtensor<double,4>& A)
{
  xt::xtensor<double,2> Am = xt::empty<double>({A.shape()[0], A.shape()[1]});
  Cartesian2d::hydrostatic(A, Am);
  return Am;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,4> Deviatoric(const xt::xtensor<double,4>& A)
{
  xt::xtensor<double,4> Ad = xt::empty<double>(A.shape());
  Cartesian2d::deviatoric(A, Ad);
  return Ad;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Epsd(const xt::xtensor<double,4>& A)
{
  xt::xtensor<double,2> Aeq = xt::empty<double>({A.shape()[0], A.shape()[1]});
  Cartesian2d::epsd(A, Aeq);
  return Aeq;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Sigd(const xt::xtensor<double,4>& A)
{
  xt::xtensor<double,2> Aeq = xt::empty<double>({A.shape()[0], A.shape()[1]});
  Cartesian2d::sigd(A, Aeq);
  return Aeq;
}

// -------------------------------------------------------------------------------------------------

}} // namespace ...

#endif
