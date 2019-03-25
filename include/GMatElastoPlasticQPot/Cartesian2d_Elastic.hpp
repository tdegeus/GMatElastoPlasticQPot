/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

================================================================================================= */

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_ELASTIC_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_ELASTIC_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

// -------------------------------------------------------------------------------------------------

inline Elastic::Elastic(double K, double G) : m_K(K), m_G(G)
{
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::K() const
{
  return m_K;
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::G() const
{
  return m_G;
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::epsp(const T2&) const
{
  return 0.0;
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::epsp(double) const
{
  return 0.0;
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::epsy(size_t) const
{
  return std::numeric_limits<double>::infinity();
}

// -------------------------------------------------------------------------------------------------

inline size_t Elastic::find(const T2&) const
{
  return 0;
}

// -------------------------------------------------------------------------------------------------

inline size_t Elastic::find(double) const
{
  return 0;
}

// -------------------------------------------------------------------------------------------------

template <class T>
inline void Elastic::stress(const T2& Eps, T&& Sig) const
{
  auto I    = Cartesian2d::I();
  auto epsm = 0.5 * trace(Eps);
  auto Epsd = Eps - epsm * I;
  xt::noalias(Sig) = m_K * epsm * I + m_G * Epsd;
}

// -------------------------------------------------------------------------------------------------

inline T2 Elastic::Stress(const T2& Eps) const
{
  T2 Sig;
  this->stress(Eps, Sig);
  return Sig;
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::energy(const T2& Eps) const
{
  auto I    = Cartesian2d::I();
  auto epsm = 0.5 * trace(Eps);
  auto Epsd = Eps - epsm * I;
  auto epsd = std::sqrt(0.5 * ddot22(Epsd,Epsd));
  auto U    = m_K * std::pow(epsm,2.0);
  auto V    = m_G * std::pow(epsd,2.0);
  return U + V;
}

// -------------------------------------------------------------------------------------------------

}} // namespace ...

#endif
