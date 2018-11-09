/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

================================================================================================= */

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_CUSP_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_CUSP_HPP

// -------------------------------------------------------------------------------------------------

#include "GMatElastoPlasticQPot.h"

// =================================================================================================

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

// -------------------------------------------------------------------------------------------------

inline Cusp::Cusp(double K, double G, const xt::xtensor<double,1> &epsy, bool init_elastic) :
  m_K(K), m_G(G)
{
  // copy sorted yield strains
  m_epsy = xt::sort(epsy);

  // extra yield strain, to force an initial elastic response
  if ( init_elastic )
    if ( m_epsy(0) != -m_epsy(1) )
      m_epsy = xt::concatenate(xt::xtuple(xt::xtensor<double,1>({-m_epsy(0)}), m_epsy));

  // check the number of yield strains
  if ( m_epsy.size() < 2 )
    throw std::runtime_error("Specify at least two yield strains 'epsy'");
}

// -------------------------------------------------------------------------------------------------

inline double Cusp::epsd(const T2s &Eps) const
{
  auto Epsd = Eps - 0.5 * trace(Eps) * eye();

  return std::sqrt(.5*ddot(Epsd,Epsd));
}

// -------------------------------------------------------------------------------------------------

inline double Cusp::epsp(const T2s &Eps) const
{
  return epsp(epsd(Eps));
}

// -------------------------------------------------------------------------------------------------

inline double Cusp::epsp(double epsd) const
{
  size_t i = find(epsd);

  return ( m_epsy(i+1) + m_epsy(i) ) / 2.;
}

// -------------------------------------------------------------------------------------------------

inline double Cusp::epsy(size_t i) const
{
  return m_epsy(i);
}

// -------------------------------------------------------------------------------------------------

inline size_t Cusp::find(const T2s &Eps) const
{
  return find(epsd(Eps));
}

// -------------------------------------------------------------------------------------------------

inline size_t Cusp::find(double epsd) const
{
  if ( epsd <= m_epsy(0) or epsd >= m_epsy(m_epsy.size()-1) )
    throw std::runtime_error("Insufficient 'epsy'");

  return std::lower_bound(m_epsy.begin(), m_epsy.end(), epsd) - m_epsy.begin() - 1;
}

// -------------------------------------------------------------------------------------------------

inline T2s Cusp::Sig(const T2s &Eps) const
{
  // decompose strain: hydrostatic part, deviatoric part
  T2s    I    = eye();
  double epsm = 0.5 * trace(Eps);
  auto   Epsd = Eps - epsm * I;
  double epsd = std::sqrt(.5*ddot(Epsd,Epsd));

  // no deviatoric strain -> only hydrostatic stress
  if ( epsd <= 0. ) return m_K * epsm * I;

  // read current yield strains
  size_t i       = find(epsd);
  double eps_min = ( m_epsy(i+1) + m_epsy(i) ) / 2.;

  // return stress tensor
  return m_K * epsm * I + m_G * (1.-eps_min/epsd) * Epsd;
}

// -------------------------------------------------------------------------------------------------

inline double Cusp::energy(const T2s &Eps) const
{
  // decompose strain: hydrostatic part, deviatoric part
  T2s    I    = eye();
  double epsm = 0.5 * trace(Eps);
  auto   Epsd = Eps - epsm * I;
  double epsd = std::sqrt(.5*ddot(Epsd,Epsd));

  // hydrostatic part of the energy
  double U = m_K * std::pow(epsm,2.);

  // read current yield strain
  size_t i       = find(epsd);
  double eps_min = ( m_epsy(i+1) + m_epsy(i) ) / 2.;
  double deps_y  = ( m_epsy(i+1) - m_epsy(i) ) / 2.;

  // deviatoric part of the energy
  double V = m_G * ( std::pow(epsd-eps_min,2.) - std::pow(deps_y,2.) );

  // return total energy
  return U + V;
}

// =================================================================================================

}} // namespace ...

// =================================================================================================

#endif
