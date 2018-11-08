/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | github.com/tdegeus/ElastoPlasticQPot

================================================================================================= */

#ifndef XELASTOPLASTICQPOT_CARTESIAN2D_ELASTIC_CPP
#define XELASTOPLASTICQPOT_CARTESIAN2D_ELASTIC_CPP

// -------------------------------------------------------------------------------------------------

#include "ElastoPlasticQPot.h"

// =================================================================================================

namespace xElastoPlasticQPot {
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

inline double Elastic::epsd(const T2s &Eps) const
{
  auto Epsd = Eps - trace(Eps)/2. * eye();

  return std::sqrt(.5*ddot(Epsd,Epsd));
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::epsp(const T2s &) const
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

inline size_t Elastic::find(const T2s &) const
{
  return 0;
}

// -------------------------------------------------------------------------------------------------

inline size_t Elastic::find(double) const
{
  return 0;
}

// -------------------------------------------------------------------------------------------------

inline T2s Elastic::Sig(const T2s &Eps) const
{
  // decompose strain: hydrostatic part, deviatoric part
  T2s  I    = eye();
  auto epsm = trace(Eps)/2.;
  auto Epsd = Eps - epsm * I;

  // return stress tensor
  return m_K * epsm * I + m_G * Epsd;
}

// -------------------------------------------------------------------------------------------------

inline double Elastic::energy(const T2s &Eps) const
{
  // decompose strain: hydrostatic part, deviatoric part
  T2s  I    = eye();
  auto epsm = trace(Eps)/2.;
  auto Epsd = Eps - epsm * I;
  auto epsd = std::sqrt(.5*ddot(Epsd,Epsd));

  // hydrostatic part of the energy
  auto U = m_K * std::pow(epsm,2.);
  // deviatoric part of the energy
  auto V = m_G * std::pow(epsd,2.);

  // return total energy
  return U + V;
}

// =================================================================================================

}} // namespace ...

// =================================================================================================

#endif
