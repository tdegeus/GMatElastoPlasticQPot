/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_SMOOTH_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_SMOOTH_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {
inline Smooth::Smooth(double K, double G, const xt::xtensor<double, 1>& epsy, bool init_elastic)
    : m_K(K), m_G(G)
{
    xt::xtensor<double, 1> y = xt::sort(epsy);

    if (init_elastic) {
        if (y(0) != -y(1)) {
            y = xt::concatenate(xt::xtuple(xt::xtensor<double, 1>({-y(0)}), y));
        }
    }

    GMATELASTOPLASTICQPOT_ASSERT(y.size() > 1);

    m_yield = QPot::Static(0.0, y);
}

inline double Smooth::K() const
{
    return m_K;
}

inline double Smooth::G() const
{
    return m_G;
}

inline xt::xtensor<double, 1> Smooth::epsy() const
{
    return m_yield.yield();
}

inline size_t Smooth::currentIndex() const
{
    return m_yield.currentIndex();
}

inline double Smooth::currentYieldLeft() const
{
    return m_yield.currentYieldLeft();
}

inline double Smooth::currentYieldRight() const
{
    return m_yield.currentYieldRight();
}

inline double Smooth::epsp() const
{
    return 0.5 * (m_yield.currentYieldLeft() + m_yield.currentYieldRight());
}

template <class T>
inline void Smooth::setStrain(const T& a)
{
    GMATELASTOPLASTICQPOT_ASSERT(detail::xtensor::has_shape(a, {2, 2}));
    return this->setStrainIterator(a.cbegin());
}

template <class T>
inline void Smooth::setStrainIterator(const T& begin)
{
    std::copy(begin, begin + 4, m_Eps.begin());

    std::array<double, 4> Epsd;
    double epsm = detail::hydrostatic_deviator(m_Eps, Epsd);
    double epsd = std::sqrt(0.5 * detail::A2_ddot_B2(Epsd, Epsd));
    m_yield.setPosition(epsd);

    m_Sig[0] = m_Sig[3] = m_K * epsm;

    if (epsd <= 0.0) {
        m_Sig[1] = m_Sig[2] = 0.0;
        return;
    }

    double eps_min = 0.5 * (m_yield.currentYieldRight() + m_yield.currentYieldLeft());
    double deps_y = 0.5 * (m_yield.currentYieldRight() - m_yield.currentYieldLeft());

    double g = (m_G / epsd) * (deps_y / M_PI) * sin(M_PI / deps_y * (epsd - eps_min));
    m_Sig[0] += g * Epsd[0];
    m_Sig[1] = g * Epsd[1];
    m_Sig[2] = g * Epsd[2];
    m_Sig[3] += g * Epsd[3];
}

template <class T>
inline void Smooth::stress(T& a) const
{
    GMATELASTOPLASTICQPOT_ASSERT(detail::xtensor::has_shape(a, {2, 2}));
    return this->stressIterator(a.begin());
}

template <class T>
inline void Smooth::stressIterator(const T& begin) const
{
    std::copy(m_Sig.begin(), m_Sig.end(), begin);
}

inline Tensor2 Smooth::Stress() const
{
    auto ret = Tensor2::from_shape({2, 2});
    this->stressIterator(ret.begin());
    return ret;
}

template <class T>
inline void Smooth::tangent(T& C) const
{
    auto II = Cartesian2d::II();
    auto I4d = Cartesian2d::I4d();
    xt::noalias(C) = 0.5 * m_K * II + m_G * I4d;
}

inline Tensor4 Smooth::Tangent() const
{
    auto ret = Tensor4::from_shape({2, 2, 2, 2});
    this->tangent(ret);
    return ret;
}

inline double Smooth::energy() const
{
    std::array<double, 4> Epsd;
    double epsm = detail::hydrostatic_deviator(m_Eps, Epsd);
    double epsd = std::sqrt(0.5 * detail::A2_ddot_B2(Epsd, Epsd));

    double U = m_K * std::pow(epsm, 2.0);

    double eps_min = 0.5 * (m_yield.currentYieldRight() + m_yield.currentYieldLeft());
    double deps_y = 0.5 * (m_yield.currentYieldRight() - m_yield.currentYieldLeft());

    double V
        = -2.0 * m_G * std::pow(deps_y / M_PI, 2.0)
        * (1.0 + cos(M_PI / deps_y * (epsd - eps_min)));

    return U + V;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
