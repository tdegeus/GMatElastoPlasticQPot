/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | github.com/tdegeus/ElastoPlasticQPot

================================================================================================= */

#ifndef XELASTOPLASTICQPOT_CARTESIAN2D_MATRIX_HPP
#define XELASTOPLASTICQPOT_CARTESIAN2D_MATRIX_HPP

// -------------------------------------------------------------------------------------------------

#include "ElastoPlasticQPot.h"

// =================================================================================================

namespace xElastoPlasticQPot {
namespace Cartesian2d {

// -------------------------------------------------------------------------------------------------

inline Matrix::Matrix(size_t nelem, size_t nip) : m_nelem(nelem), m_nip(nip)
{
  m_type  = Type::Unset * xt::ones <size_t>({m_nelem, m_nip});
  m_index =               xt::empty<size_t>({m_nelem, m_nip});
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Matrix::K() const
{
  xt::xtensor<double,2> a_K = xt::empty<double>({m_nelem, m_nip});

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < m_nelem ; ++e )
    {
      for ( size_t q = 0 ; q < m_nip ; ++q )
      {
        switch ( m_type(e,q) )
        {
          case Type::Elastic: a_K(e,q) = m_Elastic[m_index(e,q)].K(); break;
          case Type::Cusp   : a_K(e,q) = m_Cusp   [m_index(e,q)].K(); break;
          case Type::Smooth : a_K(e,q) = m_Smooth [m_index(e,q)].K(); break;
          default: std::runtime_error("Unknown material");
        }
      }
    }
  }

  return a_K;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Matrix::G() const
{
  xt::xtensor<double,2> a_G = xt::empty<double>({m_nelem, m_nip});

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < m_nelem ; ++e )
    {
      for ( size_t q = 0 ; q < m_nip ; ++q )
      {
        switch ( m_type(e,q) )
        {
          case Type::Elastic: a_G(e,q) = m_Elastic[m_index(e,q)].G(); break;
          case Type::Cusp   : a_G(e,q) = m_Cusp   [m_index(e,q)].G(); break;
          case Type::Smooth : a_G(e,q) = m_Smooth [m_index(e,q)].G(); break;
          default: std::runtime_error("Unknown material");
        }
      }
    }
  }

  return a_G;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<size_t,2> Matrix::isPlastic() const
{
  xt::xtensor<size_t,2> out = xt::zeros<size_t>({m_nelem, m_nip});

  for ( size_t e = 0 ; e < m_nelem ; ++e )
    for ( size_t q = 0 ; q < m_nip ; ++q )
      if ( m_type(e,q) != Type::Unset and m_type(e,q) != Type::Elastic )
        out(e,q) = 1;

  return out;
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::check() const
{
  for ( size_t e = 0 ; e < m_nelem ; ++e )
    for ( size_t q = 0 ; q < m_nip ; ++q )
      if ( m_type(e,q) == Type::Unset )
        throw std::runtime_error("No type set for: "+std::to_string(e)+", "+std::to_string(q));
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::setElastic(const xt::xtensor<size_t,2> &I, double K, double G)
{
  #ifndef NDEBUG

    assert( I.shape() == m_type.shape() );

    for ( size_t e = 0 ; e < m_nelem ; ++e )
      for ( size_t q = 0 ; q < m_nip ; ++q )
        if ( I(e,q) )
          assert( m_type(e,q) == Type::Unset );

  #endif

  for ( size_t e = 0 ; e < m_nelem ; ++e )
  {
    for ( size_t q = 0 ; q < m_nip ; ++q )
    {
      if ( I(e,q) )
      {
        m_type (e,q) = Type::Elastic;
        m_index(e,q) = m_Elastic.size();
      }
    }
  }

  m_Elastic.push_back(Elastic(K, G));
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::setCusp(const xt::xtensor<size_t,2> &I,
  double K, double G, const xt::xtensor<double,1> &epsy, bool init_elastic)
{
  #ifndef NDEBUG

    assert( I.shape() == m_type.shape() );

    for ( size_t e = 0 ; e < m_nelem ; ++e )
      for ( size_t q = 0 ; q < m_nip ; ++q )
        if ( I(e,q) )
          assert( m_type(e,q) == Type::Unset );

  #endif

  for ( size_t e = 0 ; e < m_nelem ; ++e )
  {
    for ( size_t q = 0 ; q < m_nip ; ++q )
    {
      if ( I(e,q) ) {
        m_type (e,q) = Type::Cusp;
        m_index(e,q) = m_Cusp.size();
      }
    }
  }

  m_Cusp.push_back(Cusp(K, G, epsy, init_elastic));
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::setSmooth(const xt::xtensor<size_t,2> &I,
  double K, double G, const xt::xtensor<double,1> &epsy, bool init_elastic)
{
  #ifndef NDEBUG

    assert( I.shape() == m_type.shape() );

    for ( size_t e = 0 ; e < m_nelem ; ++e )
      for ( size_t q = 0 ; q < m_nip ; ++q )
        if ( I(e,q) )
          assert( m_type(e,q) == Type::Unset );

  #endif

  for ( size_t e = 0 ; e < m_nelem ; ++e )
  {
    for ( size_t q = 0 ; q < m_nip ; ++q )
    {
      if ( I(e,q) ) {
        m_type (e,q) = Type::Smooth;
        m_index(e,q) = m_Smooth.size();
      }
    }
  }

  m_Smooth.push_back(Smooth(K, G, epsy, init_elastic));
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::setElastic(const xt::xtensor<size_t,2> &I, const xt::xtensor<size_t,2> &idx,
  const xt::xtensor<double,1> &K, const xt::xtensor<double,1> &G)
{
  #ifndef NDEBUG

    assert( I.shape() == m_type.shape() );
    assert( I.shape() == m_type.shape() );

    for ( size_t e = 0 ; e < m_nelem ; ++e )
      for ( size_t q = 0 ; q < m_nip ; ++q )
        if ( I(e,q) )
          assert( m_type(e,q) == Type::Unset );

    assert( xt::amax(idx)[0] == K.size()-1 );

    assert( K.size() == G.size() );

  #endif

  for ( size_t e = 0 ; e < m_nelem ; ++e )
  {
    for ( size_t q = 0 ; q < m_nip ; ++q )
    {
      if ( I(e,q) ) {
        m_type (e,q) = Type::Elastic;
        m_index(e,q) = m_Elastic.size() + idx(e,q);
      }
    }
  }

  for ( size_t i = 0 ; i < K.size() ; ++i )
    m_Elastic.push_back(Elastic(K(i), G(i)));
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::setCusp(const xt::xtensor<size_t,2> &I, const xt::xtensor<size_t,2> &idx,
  const xt::xtensor<double,1> &K, const xt::xtensor<double,1> &G,
  const xt::xtensor<double,2> &epsy, bool init_elastic)
{
  #ifndef NDEBUG

    assert( I.shape() == m_type.shape() );
    assert( I.shape() == m_type.shape() );

    for ( size_t e = 0 ; e < m_nelem ; ++e )
      for ( size_t q = 0 ; q < m_nip ; ++q )
        if ( I(e,q) )
          assert( m_type(e,q) == Type::Unset );

    assert( xt::amax(idx)[0] == K.size()-1 );

    assert( K.size() == G.size()        );
    assert( K.size() == epsy.shape()[0] );

  #endif

  for ( size_t e = 0 ; e < m_nelem ; ++e )
  {
    for ( size_t q = 0 ; q < m_nip ; ++q )
    {
      if ( I(e,q) ) {
        m_type (e,q) = Type::Cusp;
        m_index(e,q) = m_Cusp.size() + idx(e,q);
      }
    }
  }

  for ( size_t i = 0 ; i < K.size() ; ++i )
    m_Cusp.push_back(Cusp(K(i), G(i), xt::view(epsy,i,xt::all()), init_elastic));
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::setSmooth(const xt::xtensor<size_t,2> &I, const xt::xtensor<size_t,2> &idx,
  const xt::xtensor<double,1> &K, const xt::xtensor<double,1> &G,
  const xt::xtensor<double,2> &epsy, bool init_elastic)
{
  #ifndef NDEBUG

    assert( I.shape() == m_type.shape() );
    assert( I.shape() == m_type.shape() );

    for ( size_t e = 0 ; e < m_nelem ; ++e )
      for ( size_t q = 0 ; q < m_nip ; ++q )
        if ( I(e,q) )
          assert( m_type(e,q) == Type::Unset );

    assert( xt::amax(idx)[0] == K.size()-1 );

    assert( K.size() == G.size()        );
    assert( K.size() == epsy.shape()[0] );

  #endif

  for ( size_t e = 0 ; e < m_nelem ; ++e )
  {
    for ( size_t q = 0 ; q < m_nip ; ++q )
    {
      if ( I(e,q) ) {
        m_type (e,q) = Type::Smooth;
        m_index(e,q) = m_Smooth.size() + idx(e,q);
      }
    }
  }

  for ( size_t i = 0 ; i < K.size() ; ++i )
    m_Smooth.push_back(Smooth(K(i), G(i), xt::view(epsy,i,xt::all()), init_elastic));
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::Sig(const xt::xtensor<double,4> &a_Eps, xt::xtensor<double,4> &a_Sig) const
{
  assert( a_Eps.shape().size() == 4             );
  assert( a_Eps.shape()[0]     == m_nelem       );
  assert( a_Eps.shape()[1]     == m_nip         );
  assert( a_Eps.shape()[2]     == m_ndim        );
  assert( a_Eps.shape()[3]     == m_ndim        );
  assert( a_Eps.shape()        == a_Sig.shape() );

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < m_nelem ; ++e )
    {
      for ( size_t q = 0 ; q < m_nip ; ++q )
      {
        auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());
        auto Sig = xt::adapt(&a_Sig(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

        switch ( m_type(e,q) )
        {
          case Type::Elastic: xt::noalias(Sig) = m_Elastic[m_index(e,q)].Sig(Eps); break;
          case Type::Cusp   : xt::noalias(Sig) = m_Cusp   [m_index(e,q)].Sig(Eps); break;
          case Type::Smooth : xt::noalias(Sig) = m_Smooth [m_index(e,q)].Sig(Eps); break;
          default: std::runtime_error("Unknown material");
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::energy(const xt::xtensor<double,4> &a_Eps, xt::xtensor<double,2> &a_energy) const
{
  assert( a_Eps.shape().size() == 4              );
  assert( a_Eps.shape()[0]     == m_nelem        );
  assert( a_Eps.shape()[1]     == m_nip          );
  assert( a_Eps.shape()[2]     == m_ndim         );
  assert( a_Eps.shape()[3]     == m_ndim         );
  assert( a_energy.shape()     == m_type.shape() );

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < m_nelem ; ++e )
    {
      for ( size_t q = 0 ; q < m_nip ; ++q )
      {
        auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

        switch ( m_type(e,q) )
        {
          case Type::Elastic: a_energy(e,q) = m_Elastic[m_index(e,q)].energy(Eps); break;
          case Type::Cusp   : a_energy(e,q) = m_Cusp   [m_index(e,q)].energy(Eps); break;
          case Type::Smooth : a_energy(e,q) = m_Smooth [m_index(e,q)].energy(Eps); break;
          default: std::runtime_error("Unknown material");
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::find(const xt::xtensor<double,4> &a_Eps, xt::xtensor<size_t,2> &a_idx) const
{
  assert( a_Eps.shape().size() == 4              );
  assert( a_Eps.shape()[0]     == m_nelem        );
  assert( a_Eps.shape()[1]     == m_nip          );
  assert( a_Eps.shape()[2]     == m_ndim         );
  assert( a_Eps.shape()[3]     == m_ndim         );
  assert( a_idx.shape()        == m_type.shape() );

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < m_nelem ; ++e )
    {
      for ( size_t q = 0 ; q < m_nip ; ++q )
      {
        auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

        switch ( m_type(e,q) )
        {
          case Type::Elastic: a_idx(e,q) = m_Elastic[m_index(e,q)].find(Eps); break;
          case Type::Cusp   : a_idx(e,q) = m_Cusp   [m_index(e,q)].find(Eps); break;
          case Type::Smooth : a_idx(e,q) = m_Smooth [m_index(e,q)].find(Eps); break;
          default: std::runtime_error("Unknown material");
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::epsy(const xt::xtensor<size_t,2> &a_idx, xt::xtensor<double,2> &a_epsy) const
{
  assert( a_idx.shape()  == m_type.shape() );
  assert( a_epsy.shape() == m_type.shape() );

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < m_nelem ; ++e )
    {
      for ( size_t q = 0 ; q < m_nip ; ++q )
      {
        switch ( m_type(e,q) )
        {
          case Type::Elastic: a_epsy(e,q) = m_Elastic[m_index(e,q)].epsy(a_idx(e,q)); break;
          case Type::Cusp   : a_epsy(e,q) = m_Cusp   [m_index(e,q)].epsy(a_idx(e,q)); break;
          case Type::Smooth : a_epsy(e,q) = m_Smooth [m_index(e,q)].epsy(a_idx(e,q)); break;
          default: std::runtime_error("Unknown material");
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline void Matrix::epsp(const xt::xtensor<double,4> &a_Eps, xt::xtensor<double,2> &a_epsp) const
{
  assert( a_Eps.shape().size() == 4              );
  assert( a_Eps.shape()[0]     == m_nelem        );
  assert( a_Eps.shape()[1]     == m_nip          );
  assert( a_Eps.shape()[2]     == m_ndim         );
  assert( a_Eps.shape()[3]     == m_ndim         );
  assert( a_epsp.shape()       == m_type.shape() );

  #pragma omp parallel
  {
    #pragma omp for
    for ( size_t e = 0 ; e < m_nelem ; ++e )
    {
      for ( size_t q = 0 ; q < m_nip ; ++q )
      {
        auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

        switch ( m_type(e,q) )
        {
          case Type::Elastic: a_epsp(e,q) = m_Elastic[m_index(e,q)].epsp(Eps); break;
          case Type::Cusp   : a_epsp(e,q) = m_Cusp   [m_index(e,q)].epsp(Eps); break;
          case Type::Smooth : a_epsp(e,q) = m_Smooth [m_index(e,q)].epsp(Eps); break;
          default: std::runtime_error("Unknown material");
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,4> Matrix::Sig(const xt::xtensor<double,4> &a_Eps) const
{
  xt::xtensor<double,4> a_Sig = xt::empty<double>(a_Eps.shape());

  this->Sig(a_Eps, a_Sig);

  return a_Sig;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Matrix::energy(const xt::xtensor<double,4> &a_Eps) const
{
  xt::xtensor<double,2> a_energy = xt::empty<double>({m_nelem, m_nip});

  this->energy(a_Eps, a_energy);

  return a_energy;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<size_t,2> Matrix::find(const xt::xtensor<double,4> &a_Eps) const
{
  xt::xtensor<size_t,2> a_idx = xt::empty<size_t>({m_nelem, m_nip});

  this->find(a_Eps, a_idx);

  return a_idx;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Matrix::epsy(const xt::xtensor<size_t,2> &a_idx) const
{
  xt::xtensor<double,2> a_epsy = xt::empty<double>({m_nelem, m_nip});

  this->epsy(a_idx, a_epsy);

  return a_epsy;
}

// -------------------------------------------------------------------------------------------------

inline xt::xtensor<double,2> Matrix::epsp(const xt::xtensor<double,4> &a_Eps) const
{
  xt::xtensor<double,2> a_epsp = xt::empty<double>({m_nelem, m_nip});

  this->epsp(a_Eps, a_epsp);

  return a_epsp;
}

// =================================================================================================

}} // namespace ...

// =================================================================================================

#endif
