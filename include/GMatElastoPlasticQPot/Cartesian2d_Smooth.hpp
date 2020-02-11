/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_SMOOTH_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_SMOOTH_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

inline Smooth::Smooth(double K, double G, const xt::xtensor<double,1>& epsy, bool init_elastic)
    : m_K(K), m_G(G)
{
    m_epsy = xt::sort(epsy);

    if (init_elastic) {
        if (m_epsy(0) != -m_epsy(1)) {
            m_epsy = xt::concatenate(xt::xtuple(xt::xtensor<double,1>({-m_epsy(0)}), m_epsy));
        }
    }

    GMATELASTOPLASTICQPOT_ASSERT(m_epsy.size() > 1);
}

inline double Smooth::K() const
{
    return m_K;
}

inline double Smooth::G() const
{
    return m_G;
}

inline xt::xtensor<double,1> Smooth::epsy() const
{
    return m_epsy;
}

inline double Smooth::epsy(size_t i) const
{
    return m_epsy(i);
}

inline double Smooth::epsp(const Tensor2& Eps) const
{
    return this->epsp(Cartesian2d::Epsd(Eps));
}

inline double Smooth::epsp(double epsd) const
{
    size_t i = this->find(epsd);
    return 0.5 * (m_epsy(i + 1) + m_epsy(i));
}

inline size_t Smooth::find(const Tensor2& Eps) const
{
    return this->find(Cartesian2d::Epsd(Eps));
}

inline size_t Smooth::find(double epsd) const
{
    GMATELASTOPLASTICQPOT_ASSERT(epsd > m_epsy(0) && epsd < m_epsy(m_epsy.size() - 1));

    return std::lower_bound(m_epsy.begin(), m_epsy.end(), epsd) - m_epsy.begin() - 1;
}

template <class T>
inline void Smooth::stress(const Tensor2& Eps, T&& Sig) const
{
    // decompose strain: hydrostatic part, deviatoric part
    auto I = Cartesian2d::I2();
    auto epsm = 0.5 * trace(Eps);
    auto Epsd = Eps - epsm * I;
    auto epsd = std::sqrt(0.5 * A2_ddot_B2(Epsd, Epsd));

    // no deviatoric strain -> only hydrostatic stress
    if (epsd <= 0.0) {
        xt::noalias(Sig) = m_K * epsm * I;
        return;
    }

    // read current yield strains
    size_t i = this->find(epsd);
    double eps_min = 0.5 * (m_epsy(i + 1) + m_epsy(i));
    double deps_y = 0.5 * (m_epsy(i + 1) - m_epsy(i));

    // return stress tensor
    xt::noalias(Sig)
        = m_K * epsm * I
        + (m_G / epsd) * (deps_y / M_PI) * sin(M_PI / deps_y * (epsd - eps_min)) * Epsd;
}

inline Tensor2 Smooth::Stress(const Tensor2& Eps) const
{
    Tensor2 Sig;
    this->stress(Eps, Sig);
    return Sig;
}

inline double Smooth::energy(const Tensor2& Eps) const
{
    // decompose strain: hydrostatic part, deviatoric part
    auto I = Cartesian2d::I2();
    auto epsm = 0.5 * trace(Eps);
    auto Epsd = Eps - epsm * I;
    auto epsd = std::sqrt(0.5 * A2_ddot_B2(Epsd, Epsd));

    // hydrostatic part of the energy
    double U = m_K * std::pow(epsm, 2.0);

    // read current yield strain
    size_t i = this->find(epsd);
    double eps_min = 0.5 * (m_epsy(i + 1) + m_epsy(i));
    double deps_y = 0.5 * (m_epsy(i + 1) - m_epsy(i));

    // deviatoric part of the energy
    double V
        = -2.0 * m_G * std::pow(deps_y / M_PI, 2.0)
        * (1.0 + cos(M_PI / deps_y * (epsd - eps_min)));

    // return total energy
    return U + V;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
