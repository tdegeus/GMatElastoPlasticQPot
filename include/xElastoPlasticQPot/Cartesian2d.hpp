/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | github.com/tdegeus/ElastoPlasticQPot

================================================================================================= */

#ifndef XELASTOPLASTICQPOT_CARTESIAN2D_HPP
#define XELASTOPLASTICQPOT_CARTESIAN2D_HPP

// -------------------------------------------------------------------------------------------------

#include "ElastoPlasticQPot.h"

// =================================================================================================

namespace xElastoPlasticQPot {
namespace Cartesian2d {

// ======================================== TENSOR PRODUCTS ========================================

template<class T>
inline double trace(const T &A)
{
  return A(0,0) + A(1,1);
}

// -------------------------------------------------------------------------------------------------

template<class T>
inline double ddot(const T &A, const T &B)
{
  return A(0,0) * B(0,0) + 2.0 * A(0,1) * B(0,1) + A(1,1) * B(1,1);
}

// -------------------------------------------------------------------------------------------------

inline T2s eye()
{
  return T2s({{1., 0.},
              {0., 1.}});
}

// ===================================== TENSOR DECOMPOSITION ======================================

inline double epsm(const T2s &Eps)
{
  return trace(Eps)/2.;
}

// -------------------------------------------------------------------------------------------------

inline double epsd(const T2s &Eps)
{
  T2s Epsd = Eps - trace(Eps)/2. * eye();

  return std::sqrt(.5*ddot(Epsd,Epsd));
}

// -------------------------------------------------------------------------------------------------

inline T2s Epsd(const T2s &Eps)
{
  return Eps - trace(Eps)/2. * eye();
}

// -------------------------------------------------------------------------------------------------

inline double sigm(const T2s &Sig)
{
  return trace(Sig)/2.;
}

// -------------------------------------------------------------------------------------------------

inline double sigd(const T2s &Sig)
{
  T2s Sigd = Sig - trace(Sig)/2. * eye();

  return std::sqrt(2.*ddot(Sigd,Sigd));
}

// -------------------------------------------------------------------------------------------------

inline T2s Sigd(const T2s &Sig)
{
  return Sig - trace(Sig)/2. * eye();
}

// ========================= TENSOR DECOMPOSITION - MATRIX - NO ALLOCATION =========================

inline void epsm(const xt::xtensor<double,4> &a_Eps, xt::xtensor<double,2> &a_epsm)
{
  assert( a_Eps.shape()[0] == a_epsm.shape()[0] );
  assert( a_Eps.shape()[1] == a_epsm.shape()[1] );
  assert( a_Eps.shape()[2] == 2 );
  assert( a_Eps.shape()[3] == 2 );

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < a_Eps.shape()[0] ; ++e )
    {
      for ( size_t q = 0 ; q < a_Eps.shape()[1] ; ++q )
      {
        auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<2,2>());

        a_epsm(e,q) = trace(Eps)/2.;
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void epsd(const xt::xtensor<double,4> &a_Eps, xt::xtensor<double,2> &a_epsd)
{
  assert( a_Eps.shape()[0] == a_epsd.shape()[0] );
  assert( a_Eps.shape()[1] == a_epsd.shape()[1] );
  assert( a_Eps.shape()[2] == 2 );
  assert( a_Eps.shape()[3] == 2 );

  #pragma omp parallel
  {
    T2s I = eye();

    #pragma omp for
    for ( size_t e = 0 ; e < a_Eps.shape()[0] ; ++e )
    {
      for ( size_t q = 0 ; q < a_Eps.shape()[1] ; ++q )
      {
        auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<2,2>());

        auto Epsd = Eps - trace(Eps)/2. * I;

        a_epsd(e,q) = std::sqrt(.5*ddot(Epsd,Epsd));
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void Epsd(const xt::xtensor<double,4> &a_Eps, xt::xtensor<double,4> &a_Epsd)
{
  assert( a_Eps.shape()[2] == 2 );
  assert( a_Eps.shape()[3] == 2 );
  assert( a_Eps.shape() == a_Epsd.shape() );

  #pragma omp parallel
  {
    T2s I = eye();

    #pragma omp for
    for ( size_t e = 0 ; e < a_Eps.shape()[0] ; ++e )
    {
      for ( size_t q = 0 ; q < a_Eps.shape()[1] ; ++q )
      {
        auto Eps  = xt::adapt(&a_Eps (e,q,0,0), xt::xshape<2,2>());
        auto Epsd = xt::adapt(&a_Epsd(e,q,0,0), xt::xshape<2,2>());

        xt::noalias(Epsd) = Eps - trace(Eps)/2. * I;
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void sigm(const xt::xtensor<double,4> &a_Sig, xt::xtensor<double,2> &a_sigm)
{
  assert( a_Sig.shape()[0] == a_sigm.shape()[0] );
  assert( a_Sig.shape()[1] == a_sigm.shape()[1] );
  assert( a_Sig.shape()[2] == 2 );
  assert( a_Sig.shape()[3] == 2 );

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < a_Sig.shape()[0] ; ++e )
    {
      for ( size_t q = 0 ; q < a_Sig.shape()[1] ; ++q )
      {
        auto Sig = xt::adapt(&a_Sig(e,q,0,0), xt::xshape<2,2>());

        a_sigm(e,q) = trace(Sig)/2.;
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void sigd(const xt::xtensor<double,4> &a_Sig, xt::xtensor<double,2> &a_sigd)
{
  assert( a_Sig.shape()[0] == a_sigd.shape()[0] );
  assert( a_Sig.shape()[1] == a_sigd.shape()[1] );
  assert( a_Sig.shape()[2] == 2 );
  assert( a_Sig.shape()[3] == 2 );

  #pragma omp parallel
  {
    T2s I = eye();

    #pragma omp for
    for ( size_t e = 0 ; e < a_Sig.shape()[0] ; ++e )
    {
      for ( size_t q = 0 ; q < a_Sig.shape()[1] ; ++q )
      {
        auto Sig = xt::adapt(&a_Sig(e,q,0,0), xt::xshape<2,2>());

        auto Sigd = Sig - trace(Sig)/2. * I;

        a_sigd(e,q) = std::sqrt(2.*ddot(Sigd,Sigd));
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void Sigd(const xt::xtensor<double,4> &a_Sig, xt::xtensor<double,4> &a_Sigd)
{
  assert( a_Sig.shape()[2] == 2 );
  assert( a_Sig.shape()[3] == 2 );
  assert( a_Sig.shape() == a_Sigd.shape() );

  #pragma omp parallel
  {
    T2s I = eye();

    #pragma omp for
    for ( size_t e = 0 ; e < a_Sig.shape()[0] ; ++e )
    {
      for ( size_t q = 0 ; q < a_Sig.shape()[1] ; ++q )
      {
        auto Sig  = xt::adapt(&a_Sig (e,q,0,0), xt::xshape<2,2>());
        auto Sigd = xt::adapt(&a_Sigd(e,q,0,0), xt::xshape<2,2>());

        xt::noalias(Sigd) = Sig - trace(Sig)/2. * I;
      }
    }
  }
}

// ======================== TENSOR DECOMPOSITION - MATRIX - AUTO ALLOCATION ========================

inline xt::xtensor<double,2> epsm(const xt::xtensor<double,4> &a_Eps)
{
  xt::xtensor<double,2> out = xt::empty<double>({a_Eps.shape()[0], a_Eps.shape()[1]});

  epsm(a_Eps, out);

  return out;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> epsd(const xt::xtensor<double,4> &a_Eps)
{
  xt::xtensor<double,2> out = xt::empty<double>({a_Eps.shape()[0], a_Eps.shape()[1]});

  epsd(a_Eps, out);

  return out;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,4> Epsd(const xt::xtensor<double,4> &a_Eps)
{
  xt::xtensor<double,4> out = xt::empty<double>(a_Eps.shape());

  Epsd(a_Eps, out);

  return out;
}


// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> sigm(const xt::xtensor<double,4> &a_Sig)
{
  xt::xtensor<double,2> out = xt::empty<double>({a_Sig.shape()[0], a_Sig.shape()[1]});

  sigm(a_Sig, out);

  return out;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> sigd(const xt::xtensor<double,4> &a_Sig)
{
  xt::xtensor<double,2> out = xt::empty<double>({a_Sig.shape()[0], a_Sig.shape()[1]});

  sigd(a_Sig, out);

  return out;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,4> Sigd(const xt::xtensor<double,4> &a_Sig)
{
  xt::xtensor<double,4> out = xt::empty<double>(a_Sig.shape());

  Sigd(a_Sig, out);

  return out;
}

// ============================================ MAXIMUM ============================================

inline double epsm_max(const xt::xtensor<double,4> &a_Eps)
{
  assert( a_Eps.shape()[2] == 2 );
  assert( a_Eps.shape()[3] == 2 );

  // allocate maximum
  double out;

  // compute one point
  {
    auto Eps = xt::adapt(&a_Eps(0,0,0,0), xt::xshape<2,2>());

    out = trace(Eps)/2.;
  }

  // loop over all points
  for ( size_t e = 0 ; e < a_Eps.shape()[0] ; ++e )
  {
    for ( size_t q = 0 ; q < a_Eps.shape()[1] ; ++q )
    {
      auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<2,2>());

      out = std::max(out, trace(Eps)/2.);
    }
  }

  return out;
}

// -------------------------------------------------------------------------------------------------

inline double sigm_max(const xt::xtensor<double,4> &a_Sig)
{
  assert( a_Sig.shape()[2] == 2 );
  assert( a_Sig.shape()[3] == 2 );

  // allocate maximum
  double out;

  // compute one point
  {
    auto Sig = xt::adapt(&a_Sig(0,0,0,0), xt::xshape<2,2>());

    out = trace(Sig)/2.;
  }

  // loop over all points
  for ( size_t e = 0 ; e < a_Sig.shape()[0] ; ++e )
  {
    for ( size_t q = 0 ; q < a_Sig.shape()[1] ; ++q )
    {
      auto Sig = xt::adapt(&a_Sig(e,q,0,0), xt::xshape<2,2>());

      out = std::max(out, trace(Sig)/2.);
    }
  }

  return out;
}

// -------------------------------------------------------------------------------------------------

inline double epsd_max(const xt::xtensor<double,4> &a_Eps)
{
  assert( a_Eps.shape()[2] == 2 );
  assert( a_Eps.shape()[3] == 2 );

  // identity tensor
  T2s I = eye();

  // allocate maximum
  double out;

  // compute one point
  {
    auto Eps = xt::adapt(&a_Eps(0,0,0,0), xt::xshape<2,2>());

    auto Epsd = Eps - trace(Eps)/2. * I;

    out = std::sqrt(.5*ddot(Epsd,Epsd));
  }

  // loop over all points
  for ( size_t e = 0 ; e < a_Eps.shape()[0] ; ++e )
  {
    for ( size_t q = 0 ; q < a_Eps.shape()[1] ; ++q )
    {
      auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<2,2>());

      auto Epsd = Eps - trace(Eps)/2. * I;

      out = std::max(out, std::sqrt(.5*ddot(Epsd,Epsd)));
    }
  }

  return out;
}

// -------------------------------------------------------------------------------------------------

inline double sigd_max(const xt::xtensor<double,4> &a_Sig)
{
  assert( a_Sig.shape()[2] == 2 );
  assert( a_Sig.shape()[3] == 2 );

  // identity tensor
  T2s I = eye();

  // allocate maximum
  double out;

  // compute one point
  {
    auto Sig = xt::adapt(&a_Sig(0,0,0,0), xt::xshape<2,2>());

    auto Sigd = Sig - trace(Sig)/2. * I;

    out = std::sqrt(2.*ddot(Sigd,Sigd));
  }

  // loop over all points
  for ( size_t e = 0 ; e < a_Sig.shape()[0] ; ++e )
  {
    for ( size_t q = 0 ; q < a_Sig.shape()[1] ; ++q )
    {
      auto Sig = xt::adapt(&a_Sig(e,q,0,0), xt::xshape<2,2>());

      auto Sigd = Sig - trace(Sig)/2. * I;

      out = std::max(out, std::sqrt(2.*ddot(Sigd,Sigd)));
    }
  }

  return out;
}

// =================================================================================================

}} // namespace ...

// =================================================================================================

#endif
