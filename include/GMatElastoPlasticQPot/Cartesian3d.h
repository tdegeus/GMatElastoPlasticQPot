/**
\file
\copyright Copyright. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN3D_H
#define GMATELASTOPLASTICQPOT_CARTESIAN3D_H

/**
\cond
*/
// use "M_PI" from "math.h"
#define _USE_MATH_DEFINES
/**
\endcond
*/

#include <algorithm>
#include <math.h>

#include <GMatElastic/Cartesian3d.h>
#include <GMatTensor/Cartesian3d.h>
#include <QPot.h>

#include "config.h"
#include "version.h"

namespace GMatElastoPlasticQPot {

/**
Implementation in a 3-d Cartesian coordinate frame.
Note that the definitions of the bulk and shear modulus are different:
here they are identical to classical elasticity (see e.g. Landau & Lifschitz).
*/
namespace Cartesian3d {

/**
Equivalent strain: norm of strain deviator

\f$ \sqrt{\frac{1}{2} (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use epsd().

\param A [..., 3, 3] array.
\return [...] array.
*/
template <class T>
inline auto Epsd(const T& A) -> typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type
{
    GMATELASTOPLASTICQPOT_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 2) == 3);

    using return_type = typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type;
    return_type ret = GMatTensor::Cartesian3d::Norm_deviatoric(A);
    ret *= std::sqrt(0.5);
    return ret;
}

/**
Same as Epsd(), but writes to externally allocated output.

\param A [..., 3, 3] array.
\param ret output [...] array
*/
template <class T, class U>
inline void epsd(const T& A, U& ret)
{
    GMATELASTOPLASTICQPOT_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 2) == 3);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, ret.shape()));

    GMatTensor::Cartesian3d::norm_deviatoric(A, ret);
    ret *= std::sqrt(0.5);
}

/**
Equivalent stress: norm of strain deviator

\f$ \sqrt{2 (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use sigd().

\param A [..., 3, 3] array.
\return [...] array.
*/
template <class T>
inline auto Sigd(const T& A) -> typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type
{
    GMATELASTOPLASTICQPOT_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 2) == 3);

    using return_type = typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type;
    return_type ret = GMatTensor::Cartesian3d::Norm_deviatoric(A);
    ret *= std::sqrt(2.0);
    return ret;
}

/**
Same as Sigd(), but writes to externally allocated output.

\param A [..., 3, 3] array.
\param ret output [...] array
*/
template <class T, class U>
inline void sigd(const T& A, U& ret)
{
    GMATELASTOPLASTICQPOT_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape(A.dimension() - 2) == 3);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, ret.shape()));

    GMatTensor::Cartesian3d::norm_deviatoric(A, ret);
    ret *= std::sqrt(2.0);
}

/**
Array of material points with an elasto-plastic material model.
The response is defined by a potential energy landscape consisting of a sequence of parabolic wells.

The corresponding elastic response is GMatElastic::Cartesian3d::Elastic.

\tparam N Rank of the array.
*/
template <size_t N>
class Cusp : public GMatElastic::Cartesian3d::Elastic<N> {
protected:
    array_type::tensor<size_t, N> m_i; ///< Index of the current yield strain per item.
    array_type::tensor<double, N> m_eps; ///< Equivalent strain.
    array_type::tensor<double, N + 1> m_epsy; ///< Yield strain sequence.
    size_t m_nyield; ///< shape(-1)

    using GMatElastic::Cartesian3d::Elastic<N>::m_K;
    using GMatElastic::Cartesian3d::Elastic<N>::m_G;
    using GMatElastic::Cartesian3d::Elastic<N>::m_Eps;
    using GMatElastic::Cartesian3d::Elastic<N>::m_Sig;
    using GMatElastic::Cartesian3d::Elastic<N>::m_C;

    using GMatTensor::Cartesian3d::Array<N>::m_ndim;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian3d::Array<N>::m_size;
    using GMatTensor::Cartesian3d::Array<N>::m_shape;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian3d::Array<N>::rank;

    Cusp() = default;

    /**
    Construct system.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    \param epsy Yield strain sequence per item.
    */
    template <class T, class Y>
    Cusp(const T& K, const T& G, const Y& epsy)
    {
        this->init_Cusp(K, G, epsy);
    }

protected:
    /**
    Construct system.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    \param epsy Yield strain sequence per item.
    */
    template <class T, class Y>
    void init_Cusp(const T& K, const T& G, const Y& epsy)
    {
        this->init_Elastic(K, G);
        m_eps = xt::zeros<double>(m_shape);
        m_i = xt::zeros<size_t>(m_shape);
        this->set_epsy(epsy);
    }

public:
    /**
    Yield strains per item.
    \return [shape()].
    */
    const array_type::tensor<double, N + 1>& epsy() const
    {
        return m_epsy;
    }

    /**
    Overwrite yield strains per item.
    \param epsy Yield strain sequence per item.
    */
    template <class T>
    void set_epsy(const T& epsy)
    {

        GMATELASTOPLASTICQPOT_ASSERT(epsy.dimension() == N + 1);
        GMATELASTOPLASTICQPOT_ASSERT(
            std::equal(m_shape.cbegin(), m_shape.cend(), epsy.shape().cbegin()));

        m_epsy = epsy;
        m_nyield = m_epsy.shape(N);

#ifdef GMATELASTOPLASTICQPOT_ENABLE_ASSERT
        for (size_t i = 0; i < m_size; ++i) {
            double* y = &m_epsy.flat(i * m_nyield);
            GMATELASTOPLASTICQPOT_ASSERT(std::is_sorted(y, y + m_nyield));
        }
#endif

        this->refresh();
    }

    /**
    Index of the current yield strain per item.
    By definition `epsy[..., i] < eps[...] <= epsy[..., i + 1]`
    \return [shape()].
    */
    const array_type::tensor<size_t, N>& i() const
    {
        return m_i;
    }

    /**
    Equivalent strain per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& eps() const
    {
        return m_eps;
    }

    /**
    Current yield strain left per item.
    Convenience function, same as `epsy[..., i]`.
    \return [shape()].
    */
    array_type::tensor<double, N> epsy_left() const
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.flat(i) = m_epsy.flat(i * m_nyield + m_i.flat(i));
        }

        return ret;
    }

    /**
    Current yield strain right per item.
    Convenience function, same as `epsy[..., i + 1]`.
    \return [shape()].
    */
    array_type::tensor<double, N> epsy_right() const
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            ret.flat(i) = m_epsy.flat(i * m_nyield + m_i.flat(i) + 1);
        }

        return ret;
    }

    /**
    Plastic strain per item.
    Convenience function, same as `(epsy[..., i] + epsy[..., i + 1]) / 2`.
    \return [shape()].
    */
    array_type::tensor<double, N> epsp() const
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);

        for (size_t i = 0; i < m_size; ++i) {
            auto* y = &m_epsy.flat(i * m_nyield + m_i.flat(i));
            ret.flat(i) = 0.5 * (*(y) + *(y + 1));
        }

        return ret;
    }

    void refresh(bool compute_tangent = true) override
    {
        (void)(compute_tangent);

        namespace GT = GMatTensor::Cartesian3d::pointer;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {

            double K = m_K.flat(i);
            double G = m_G.flat(i);

            const double* Eps = &m_Eps.flat(i * m_stride_tensor2);
            double* Sig = &m_Sig.flat(i * m_stride_tensor2);

            std::array<double, m_stride_tensor2> Epsd;
            double epsm = GT::Hydrostatic_deviatoric(Eps, &Epsd[0]);
            double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));
            m_eps.flat(i) = epsd;

            const double* y = &m_epsy.flat(i * m_nyield);
            size_t idx = QPot::iterator::lower_bound(y, y + m_nyield, epsd, m_i.flat(i));
            m_i.flat(i) = idx;

            Sig[0] = Sig[4] = Sig[8] = 3.0 * K * epsm;

            if (epsd <= 0.0) {
                Sig[1] = Sig[2] = Sig[3] = Sig[5] = Sig[6] = Sig[7] = 0.0;
                continue;
            }

            double eps_min = 0.5 * (*(y + idx) + *(y + idx + 1));

            double g = 2.0 * G * (1.0 - eps_min / epsd);
            Sig[0] += g * Epsd[0];
            Sig[1] = g * Epsd[1];
            Sig[2] = g * Epsd[2];
            Sig[3] = g * Epsd[3];
            Sig[4] += g * Epsd[4];
            Sig[5] = g * Epsd[5];
            Sig[6] = g * Epsd[6];
            Sig[7] = g * Epsd[7];
            Sig[8] += g * Epsd[8];
        }
    }

    array_type::tensor<double, N> energy() const override
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);
        namespace GT = GMatTensor::Cartesian3d::pointer;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {

            double K = m_K.flat(i);
            double G = m_G.flat(i);

            const double* Eps = &m_Eps.flat(i * m_stride_tensor2);

            std::array<double, m_stride_tensor2> Epsd;
            double epsm = GT::Hydrostatic_deviatoric(Eps, &Epsd[0]);
            double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));

            const double* y = &m_epsy.flat(i * m_nyield);
            size_t idx = m_i.flat(i);

            double eps_min = 0.5 * (*(y + idx) + *(y + idx + 1));
            double deps_y = 0.5 * (*(y + idx + 1) - *(y + idx));

            double U = 3.0 * K * std::pow(epsm, 2.0);
            double V = 2.0 * G * (std::pow(epsd - eps_min, 2.0) - std::pow(deps_y, 2.0));

            ret.flat(i) = U + V;
        }

        return ret;
    }
};

/**
Array of material points with an elasto-plastic material model.
The response is defined by a potential energy landscape consisting of a sequence of smooth wells.

The corresponding elastic response is GMatElastic::Cartesian3d::Elastic.

\tparam N Rank of the array.
*/
template <size_t N>
class Smooth : public Cusp<N> {
protected:
    using Cusp<N>::m_i;
    using Cusp<N>::m_eps;
    using Cusp<N>::m_epsy;
    using Cusp<N>::m_nyield;

    using GMatElastic::Cartesian3d::Elastic<N>::m_K;
    using GMatElastic::Cartesian3d::Elastic<N>::m_G;
    using GMatElastic::Cartesian3d::Elastic<N>::m_Eps;
    using GMatElastic::Cartesian3d::Elastic<N>::m_Sig;
    using GMatElastic::Cartesian3d::Elastic<N>::m_C;

    using GMatTensor::Cartesian3d::Array<N>::m_ndim;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian3d::Array<N>::m_size;
    using GMatTensor::Cartesian3d::Array<N>::m_shape;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian3d::Array<N>::rank;

    Smooth() = default;

    /**
    Construct system.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    \param epsy Yield strain sequence per item.
    */
    template <class T, class Y>
    Smooth(const T& K, const T& G, const Y& epsy)
    {
        this->init_Cusp(K, G, epsy);
    }

    void refresh(bool compute_tangent = true) override
    {
        (void)(compute_tangent);

        namespace GT = GMatTensor::Cartesian3d::pointer;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {

            double K = m_K.flat(i);
            double G = m_G.flat(i);

            const double* Eps = &m_Eps.flat(i * m_stride_tensor2);
            double* Sig = &m_Sig.flat(i * m_stride_tensor2);

            std::array<double, m_stride_tensor2> Epsd;
            double epsm = GT::Hydrostatic_deviatoric(Eps, &Epsd[0]);
            double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));
            m_eps.flat(i) = epsd;

            const double* y = &m_epsy.flat(i * m_nyield);
            size_t idx = QPot::iterator::lower_bound(y, y + m_nyield, epsd, m_i.flat(i));
            m_i.flat(i) = idx;

            Sig[0] = Sig[4] = Sig[8] = 3.0 * K * epsm;

            if (epsd <= 0.0) {
                Sig[1] = Sig[2] = Sig[3] = Sig[5] = Sig[6] = Sig[7] = 0.0;
                continue;
            }

            double eps_min = 0.5 * (*(y + idx) + *(y + idx + 1));
            double deps_y = 0.5 * (*(y + idx + 1) - *(y + idx));

            double g = (2.0 * G / epsd) * (deps_y / M_PI) * sin(M_PI / deps_y * (epsd - eps_min));
            Sig[0] += g * Epsd[0];
            Sig[1] = g * Epsd[1];
            Sig[2] = g * Epsd[2];
            Sig[3] = g * Epsd[3];
            Sig[4] += g * Epsd[4];
            Sig[5] = g * Epsd[5];
            Sig[6] = g * Epsd[6];
            Sig[7] = g * Epsd[7];
            Sig[8] += g * Epsd[8];
        }
    }

    array_type::tensor<double, N> energy() const override
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);
        namespace GT = GMatTensor::Cartesian3d::pointer;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {

            double K = m_K.flat(i);
            double G = m_G.flat(i);

            const double* Eps = &m_Eps.flat(i * m_stride_tensor2);

            std::array<double, m_stride_tensor2> Epsd;
            double epsm = GT::Hydrostatic_deviatoric(Eps, &Epsd[0]);
            double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));

            const double* y = &m_epsy.flat(i * m_nyield);
            size_t idx = m_i.flat(i);

            double eps_min = 0.5 * (*(y + idx) + *(y + idx + 1));
            double deps_y = 0.5 * (*(y + idx + 1) - *(y + idx));

            double U = 3.0 * K * std::pow(epsm, 2.0);
            double V = -4.0 * G * std::pow(deps_y / M_PI, 2.0) *
                       (1.0 + cos(M_PI / deps_y * (epsd - eps_min)));

            ret.flat(i) = U + V;
        }

        return ret;
    }
};

} // namespace Cartesian3d
} // namespace GMatElastoPlasticQPot

#endif
