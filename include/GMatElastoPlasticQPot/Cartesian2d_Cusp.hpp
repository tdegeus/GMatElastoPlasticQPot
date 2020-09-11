/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_CUSP_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_CUSP_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

inline Cusp::Cusp(double K, double G, const xt::xtensor<double,1>& epsy, bool init_elastic)
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

inline double Cusp::K() const
{
    return m_K;
}

inline double Cusp::G() const
{
    return m_G;
}

inline xt::xtensor<double,1> Cusp::epsy() const
{
    return m_epsy;
}

inline double Cusp::epsy(size_t i) const
{
    return m_epsy(i);
}

inline double Cusp::epsp(const Tensor2& Eps) const
{
    return this->epsp(Cartesian2d::Epsd(Eps));
}

inline double Cusp::epsp(double epsd) const
{
    size_t i = this->find(epsd);
    return 0.5 * (m_epsy(i + 1) + m_epsy(i));
}

inline size_t Cusp::find(const Tensor2& Eps) const
{
    return this->find(Cartesian2d::Epsd(Eps));
}

inline size_t Cusp::find(double epsd) const
{
    GMATELASTOPLASTICQPOT_ASSERT(epsd > m_epsy(0) && epsd < m_epsy(m_epsy.size() - 1));

    return std::lower_bound(m_epsy.begin(), m_epsy.end(), epsd) - m_epsy.begin() - 1;
}

template <class T>
inline void Cusp::stress(const Tensor2& Eps_in, T&& Sig_out) const
{
    std::array<double,4> Eps;
    std::array<double,4> Epsd;
    std::array<double,4> Sig;
    std::copy(Eps_in.begin(), Eps_in.end(), Eps.begin());
    double epsm = 0.5 * trace_new(Eps);
    deviator(Eps, epsm, Epsd);
    double epsd = std::sqrt(0.5 * A2_ddot_B2_sym(Epsd, Epsd));
    Sig[0] = Sig[3] = m_K * epsm;

    // no deviatoric strain -> only hydrostatic stress
    if (epsd <= 0.0) {
        Sig[1] = Sig[2] = 0.0;
        std::copy(Sig.begin(), Sig.end(), Sig_out.begin());
        return;
    }

    // read current yield strains
    size_t i = this->find(epsd);
    double eps_min = 0.5 * (m_epsy(i + 1) + m_epsy(i));

    // return stress tensor
    // xt::noalias(Sig) = m_K * epsm * I + m_G * (1.0 - eps_min / epsd) * Epsd;
    Sig[0] += m_G * (1.0 - eps_min / epsd) * Epsd[0];
    Sig[1] = m_G * (1.0 - eps_min / epsd) * Epsd[1];
    Sig[2] = m_G * (1.0 - eps_min / epsd) * Epsd[2];
    Sig[3] += m_G * (1.0 - eps_min / epsd) * Epsd[3];
    std::copy(Sig.begin(), Sig.end(), Sig_out.begin());
}

inline Tensor2 Cusp::Stress(const Tensor2& Eps) const
{
    Tensor2 Sig;
    this->stress(Eps, Sig);
    return Sig;
}

template <class T, class S>
inline void Cusp::tangent(const Tensor2& Eps, T&& Sig, S&& C) const
{
    auto II = Cartesian2d::II();
    auto I4d = Cartesian2d::I4d();
    this->stress(Eps, Sig);
    xt::noalias(C) = 0.5 * m_K * II + m_G * I4d;
}

inline std::tuple<Tensor2, Tensor4> Cusp::Tangent(const Tensor2& Eps) const
{
    Tensor2 Sig;
    Tensor4 C;
    this->tangent(Eps, Sig, C);
    return std::make_tuple(Sig, C);
}

inline double Cusp::energy(const Tensor2& Eps) const
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
    double V = m_G * (std::pow(epsd - eps_min, 2.0) - std::pow(deps_y, 2.0));

    // return total energy
    return U + V;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
