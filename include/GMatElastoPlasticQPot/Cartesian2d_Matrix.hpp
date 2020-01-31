/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_MATRIX_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_MATRIX_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {


inline Matrix::Matrix(size_t nelem, size_t nip) : m_nelem(nelem), m_nip(nip)
{
    m_type = xt::ones<size_t>({m_nelem, m_nip}) * Type::Unset;
    m_index = xt::empty<size_t>({m_nelem, m_nip});
    m_allSet = false;
}


inline size_t Matrix::ndim() const
{
    return m_ndim;
}


inline size_t Matrix::nelem() const
{
    return m_nelem;
}


inline size_t Matrix::nip() const
{
    return m_nip;
}


inline xt::xtensor<size_t,2> Matrix::type() const
{
    return m_type;
}


inline xt::xtensor<double,2> Matrix::K() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);

    xt::xtensor<double,2> out = xt::empty<double>({m_nelem, m_nip});

    #pragma omp parallel for
    for (size_t e = 0; e < m_nelem; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            switch (m_type(e,q)) {
                case Type::Elastic:
                    out(e,q) = m_Elastic[m_index(e,q)].K();
                    break;
                case Type::Cusp:
                    out(e,q) = m_Cusp[m_index(e,q)].K();
                    break;
                case Type::Smooth:
                    out(e,q) = m_Smooth[m_index(e,q)].K();
                    break;
            }
        }
    }

    return out;
}


inline xt::xtensor<double,2> Matrix::G() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);

    xt::xtensor<double,2> out = xt::empty<double>({m_nelem, m_nip});

    #pragma omp parallel for
    for (size_t e = 0; e < m_nelem; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            switch (m_type(e,q)) {
                case Type::Elastic:
                    out(e,q) = m_Elastic[m_index(e,q)].G();
                    break;
                case Type::Cusp:
                    out(e,q) = m_Cusp[m_index(e,q)].G();
                    break;
                case Type::Smooth:
                    out(e,q) = m_Smooth[m_index(e,q)].G();
                    break;
            }
        }
    }

    return out;
}


inline xt::xtensor<size_t,2> Matrix::isPlastic() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);

    xt::xtensor<size_t,2> out = xt::where(xt::not_equal(m_type, Type::Elastic), 1ul, 0ul);
    return out;
}


inline void Matrix::check() const
{
    if (xt::any(xt::equal(m_type, Type::Unset))) {
        throw std::runtime_error("Points without material found");
    }
}


inline void Matrix::checkAllSet()
{
    if (xt::any(xt::equal(m_type, Type::Unset))) {
        m_allSet = false;
    }
    else {
        m_allSet = true;
    }
}


inline void Matrix::setElastic(const xt::xtensor<size_t,2>& I, double K, double G)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I,0ul) || xt::equal(I,1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(\
        xt::all(xt::equal(xt::where(xt::equal(I,1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Elastic, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Elastic.size(), m_index);
    this->checkAllSet();
    m_Elastic.push_back(Elastic(K, G));
}


inline void Matrix::setCusp(
        const xt::xtensor<size_t,2>& I,
        double K,
        double G,
        const xt::xtensor<double,1>& epsy,
        bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I,0ul) || xt::equal(I,1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(\
        xt::all(xt::equal(xt::where(xt::equal(I,1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Cusp, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Cusp.size(), m_index);
    this->checkAllSet();
    m_Cusp.push_back(Cusp(K, G, epsy, init_elastic));
}


inline void Matrix::setSmooth(
        const xt::xtensor<size_t,2>& I,
        double K,
        double G,
        const xt::xtensor<double,1>& epsy,
        bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I,0ul) || xt::equal(I,1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(\
        xt::all(xt::equal(xt::where(xt::equal(I,1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Smooth, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Smooth.size(), m_index);
    this->checkAllSet();
    m_Smooth.push_back(Smooth(K, G, epsy, init_elastic));
}


inline void Matrix::setElastic(
        const xt::xtensor<size_t,2>& I,
        const xt::xtensor<size_t,2>& idx,
        const xt::xtensor<double,1>& K,
        const xt::xtensor<double,1>& G)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)[0] == K.size()-1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I,0ul) || xt::equal(I,1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(\
        xt::all(xt::equal(xt::where(xt::equal(I,1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Elastic, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Elastic.size() + idx, m_index);
    this->checkAllSet();

    for (size_t i = 0; i < K.size(); ++i) {
        m_Elastic.push_back(Elastic(K(i), G(i)));
    }
}


inline void Matrix::setCusp(
        const xt::xtensor<size_t,2>& I,
        const xt::xtensor<size_t,2>& idx,
        const xt::xtensor<double,1>& K,
        const xt::xtensor<double,1>& G,
        const xt::xtensor<double,2>& epsy,
        bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)[0] == K.size()-1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape()[0]);
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I,0ul) || xt::equal(I,1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(\
        xt::all(xt::equal(xt::where(xt::equal(I,1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Cusp, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Cusp.size() + idx, m_index);
    this->checkAllSet();

    for (size_t i = 0; i < K.size(); ++i) {
        m_Cusp.push_back(Cusp(K(i), G(i), xt::view(epsy,i,xt::all()), init_elastic));
    }
}


inline void Matrix::setSmooth(
  const xt::xtensor<size_t,2>& I,
  const xt::xtensor<size_t,2>& idx,
  const xt::xtensor<double,1>& K,
  const xt::xtensor<double,1>& G,
  const xt::xtensor<double,2>& epsy,
  bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)[0] == K.size()-1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape()[0]);
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I,0ul) || xt::equal(I,1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(\
        xt::all(xt::equal(xt::where(xt::equal(I,1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Smooth, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Smooth.size() + idx, m_index);
    this->checkAllSet();

    for (size_t i = 0; i < K.size(); ++i) {
        m_Smooth.push_back(Smooth(K(i), G(i), xt::view(epsy,i,xt::all()), init_elastic));
    }
}


inline void Matrix::stress(const xt::xtensor<double,4>& a_Eps, xt::xtensor<double,4>& a_Sig) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(a_Eps.shape() ==\
        std::decay_t<decltype(a_Eps)>::shape_type({m_nelem, m_nip, m_ndim, m_ndim}));
    GMATELASTOPLASTICQPOT_ASSERT(a_Eps.shape() == a_Sig.shape());

    #pragma omp parallel for
    for (size_t e = 0; e < m_nelem; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());
            auto Sig = xt::adapt(&a_Sig(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

            switch (m_type(e,q)) {
                case Type::Elastic:
                    m_Elastic[m_index(e,q)].stress(Eps, Sig);
                    break;
                case Type::Cusp:
                    m_Cusp[m_index(e,q)].stress(Eps, Sig);
                    break;
                case Type::Smooth:
                    m_Smooth[m_index(e,q)].stress(Eps, Sig);
                    break;
            }
        }
    }
}


inline void Matrix::energy(const xt::xtensor<double,4>& a_Eps, xt::xtensor<double,2>& a_energy) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(a_Eps.shape() ==\
        std::decay_t<decltype(a_Eps)>::shape_type({m_nelem, m_nip, m_ndim, m_ndim}));
    GMATELASTOPLASTICQPOT_ASSERT(a_energy.shape() == m_type.shape());

    #pragma omp parallel for
    for (size_t e = 0; e < m_nelem; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

            switch (m_type(e,q)) {
                case Type::Elastic:
                    a_energy(e,q) = m_Elastic[m_index(e,q)].energy(Eps);
                    break;
                case Type::Cusp:
                    a_energy(e,q) = m_Cusp[m_index(e,q)].energy(Eps);
                    break;
                case Type::Smooth:
                    a_energy(e,q) = m_Smooth[m_index(e,q)].energy(Eps);
                    break;
            }
        }
    }
}


inline void Matrix::find(const xt::xtensor<double,4>& a_Eps, xt::xtensor<size_t,2>& a_idx) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(a_Eps.shape() ==\
        std::decay_t<decltype(a_Eps)>::shape_type({m_nelem, m_nip, m_ndim, m_ndim}));
    GMATELASTOPLASTICQPOT_ASSERT(a_idx.shape() == m_type.shape());

    #pragma omp parallel for
    for (size_t e = 0; e < m_nelem; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

            switch (m_type(e,q)) {
                case Type::Elastic:
                    a_idx(e,q) = 0;
                    break;
                case Type::Cusp:
                    a_idx(e,q) = m_Cusp[m_index(e,q)].find(Eps);
                    break;
                case Type::Smooth:
                    a_idx(e,q) = m_Smooth[m_index(e,q)].find(Eps);
                    break;
            }
        }
    }
}


inline void Matrix::epsy(const xt::xtensor<size_t,2>& a_idx, xt::xtensor<double,2>& a_epsy) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(a_idx.shape() == m_type.shape());
    GMATELASTOPLASTICQPOT_ASSERT(a_epsy.shape() == m_type.shape());

    #pragma omp parallel for
    for (size_t e = 0; e < m_nelem; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            switch (m_type(e,q)) {
              case Type::Elastic:
                  a_epsy(e,q) = std::numeric_limits<double>::infinity();
                  break;
              case Type::Cusp:
                  a_epsy(e,q) = m_Cusp[m_index(e,q)].epsy(a_idx(e,q));
                  break;
              case Type::Smooth:
                  a_epsy(e,q) = m_Smooth[m_index(e,q)].epsy(a_idx(e,q));
                  break;
            }
        }
    }
}


inline void Matrix::epsp(const xt::xtensor<double,4>& a_Eps, xt::xtensor<double,2>& a_epsp) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(a_Eps.shape() ==\
        std::decay_t<decltype(a_Eps)>::shape_type({m_nelem, m_nip, m_ndim, m_ndim}));
    GMATELASTOPLASTICQPOT_ASSERT(a_epsp.shape() == m_type.shape());

    #pragma omp parallel for
    for (size_t e = 0; e < m_nelem; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {

            auto Eps = xt::adapt(&a_Eps(e,q,0,0), xt::xshape<m_ndim,m_ndim>());

            switch (m_type(e,q)) {
              case Type::Elastic:
                  a_epsp(e,q) = 0.0;
                  break;
              case Type::Cusp:
                  a_epsp(e,q) = m_Cusp[m_index(e,q)].epsp(Eps);
                  break;
              case Type::Smooth:
                  a_epsp(e,q) = m_Smooth[m_index(e,q)].epsp(Eps);
                  break;
            }
        }
    }
}


inline xt::xtensor<double,4> Matrix::Stress(const xt::xtensor<double,4>& Eps) const
{
    xt::xtensor<double,4> Sig = xt::empty<double>(Eps.shape());
    this->stress(Eps, Sig);
    return Sig;
}


inline xt::xtensor<double,2> Matrix::Energy(const xt::xtensor<double,4>& Eps) const
{
    xt::xtensor<double,2> out = xt::empty<double>({m_nelem, m_nip});
    this->energy(Eps, out);
    return out;
}


inline xt::xtensor<size_t,2> Matrix::Find(const xt::xtensor<double,4>& Eps) const
{
    xt::xtensor<size_t,2> out = xt::empty<size_t>({m_nelem, m_nip});
    this->find(Eps, out);
    return out;
}


inline xt::xtensor<double,2> Matrix::Epsy(const xt::xtensor<size_t,2>& idx) const
{
    xt::xtensor<double,2> out = xt::empty<double>({m_nelem, m_nip});
    this->epsy(idx, out);
    return out;
}


inline xt::xtensor<double,2> Matrix::Epsp(const xt::xtensor<double,4>& Eps) const
{
    xt::xtensor<double,2> out = xt::empty<double>({m_nelem, m_nip});
    this->epsp(Eps, out);
    return out;
}


}} // namespace ...


#endif
