/**
\file
\copyright Copyright. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_H
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_H

// use "M_PI" from "math.h"
#define _USE_MATH_DEFINES

#include <algorithm>
#include <math.h>

#include <GMatTensor/Cartesian2d.h>
#include <QPot.h>

#include "config.h"
#include "version.h"

namespace GMatElastoPlasticQPot {

/**
Implementation in a 2-d Cartesian coordinate frame.
*/
namespace Cartesian2d {

/**
Equivalent strain: norm of strain deviator

\f$ \sqrt{\frac{1}{2} (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use epsd().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Epsd(const T& A) -> typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type
{
    return xt::eval(std::sqrt(0.5) * GMatTensor::Cartesian2d::Norm_deviatoric(A));
}

/**
Same as Epsd(), but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array
*/
template <class T, class U>
inline void epsd(const T& A, U& ret)
{
    GMatTensor::Cartesian2d::norm_deviatoric(A, ret);
    ret *= std::sqrt(0.5);
}

/**
Equivalent stress: norm of strain deviator

\f$ \sqrt{2 (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use sigd().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Sigd(const T& A) -> typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type
{
    return xt::eval(std::sqrt(2.0) * GMatTensor::Cartesian2d::Norm_deviatoric(A));
}

/**
Same as Sigd(), but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array
*/
template <class T, class U>
inline void sigd(const T& A, U& ret)
{
    GMatTensor::Cartesian2d::norm_deviatoric(A, ret);
    ret *= std::sqrt(2.0);
}

/**
Array of material points with a linear elastic constitutive response.
\tparam N Rank of the array.
*/
template <size_t N>
class Elastic : public GMatTensor::Cartesian2d::Array<N> {
protected:
    array_type::tensor<double, N> m_K; ///< Bulk modulus per item.
    array_type::tensor<double, N> m_G; ///< Shear modulus per item.
    array_type::tensor<double, N + 2> m_Eps; ///< Strain tensor per item.
    array_type::tensor<double, N + 2> m_Sig; ///< Stress tensor per item.
    array_type::tensor<double, N + 4> m_C; ///< Tangent per item.

    using GMatTensor::Cartesian2d::Array<N>::m_ndim;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian2d::Array<N>::rank;

    Elastic() = default;

    /**
    Construct system.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    */
    template <class T>
    Elastic(const T& K, const T& G)
    {
        this->init_Elastic(K, G);
    }

protected:
    /**
    Constructor alias.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    */
    template <class T>
    void init_Elastic(const T& K, const T& G)
    {
        GMATELASTOPLASTICQPOT_ASSERT(K.dimension() == N);
        GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(K, G.shape()));
        std::copy(K.shape().cbegin(), K.shape().cend(), m_shape.begin());
        this->init(m_shape);

        m_K = K;
        m_G = G;
        m_Eps = xt::zeros<double>(m_shape_tensor2);
        m_Sig = xt::zeros<double>(m_shape_tensor2);
        m_C = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel
        {
            auto C = xt::adapt(m_C.data(), {m_ndim, m_ndim, m_ndim, m_ndim});
            double K;
            double G;
            auto II = GMatTensor::Cartesian2d::II();
            auto I4d = GMatTensor::Cartesian2d::I4d();

#pragma omp for
            for (size_t i = 0; i < m_size; ++i) {
                C.reset_buffer(&m_C.flat(i * m_stride_tensor4), m_stride_tensor4);
                K = m_K.flat(i);
                G = m_G.flat(i);
                C = 0.5 * K * II + G * I4d;
            }
        }
    }

public:
    /**
    Bulk modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& K() const
    {
        return m_K;
    }

    /**
    Shear modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& G() const
    {
        return m_G;
    }

    /**
    Set strain tensors.
    Internally, this calls refresh() to update stress.
    \tparam T e.g. `array_type::tensor<double, N + 2>`
    \param arg Strain tensor per item [shape(), 3, 3].
    */
    template <class T>
    void set_Eps(const T& arg)
    {
        GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_Eps.begin());
        this->refresh();
    }

    /**
    Recompute stress from strain.

    From C++, this function need **never** be called: the API takes care of this.

    For Python, this function should **only** be called when you modify elements of Eps().
    For example

        mat.Eps[e, q, 0, 1] = value
        ...
        mat.refresh() # "Eps" was changed without "mat" knowing

    Instead, if you write an nd-array, the API takes care of the refresh. I.e.

        mat.Eps = new_Eps
        # no further action needed, "mat" was refreshed

    Note though that you can call this function as often as you like, you will only loose time.
    */
    virtual void refresh()
    {
#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {

            double K = m_K.flat(i);
            double G = m_G.flat(i);

            const double* Eps = &m_Eps.flat(i * m_stride_tensor2);
            double* Sig = &m_Sig.flat(i * m_stride_tensor2);

            double epsm = GMatTensor::Cartesian2d::pointer::Hydrostatic(Eps);

            Sig[0] = (K - G) * epsm + G * Eps[0];
            Sig[1] = G * Eps[1];
            Sig[2] = G * Eps[2];
            Sig[3] = (K - G) * epsm + G * Eps[3];
        }
    }

    /**
    Strain tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& Eps() const
    {
        return m_Eps;
    }

    /**
    Strain tensor per item.
    The user is responsible for calling refresh() after modifying entries.
    \return [shape(), 3, 3].
    */
    array_type::tensor<double, N + 2>& Eps()
    {
        return m_Eps;
    }

    /**
    Stress tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& Sig() const
    {
        return m_Sig;
    }

    /**
    Tangent tensor per item.
    \return [shape(), 3, 3, 3, 3].
    */
    const array_type::tensor<double, N + 4>& C() const
    {
        return m_C;
    }

    /**
    Potential energy per item.
    \return [shape()].
    */
    virtual array_type::tensor<double, N> energy() const
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);
        namespace GT = GMatTensor::Cartesian2d::pointer;

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {

            double K = m_K.flat(i);
            double G = m_G.flat(i);

            const double* Eps = &m_Eps.flat(i * m_stride_tensor2);

            std::array<double, m_stride_tensor2> Epsd;
            double epsm = GT::Hydrostatic_deviatoric(Eps, &Epsd[0]);
            double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));

            double U = K * std::pow(epsm, 2.0);
            double V = G * std::pow(epsd, 2.0);

            ret.flat(i) = U + V;
        }

        return ret;
    }
};

/**
Array of material points with an elasto-plastic material model.
The response is defined by a potential energy landscape consisting of a sequence of parabolic wells.
\tparam N Rank of the array.
*/
template <size_t N>
class Cusp : public Elastic<N> {
protected:
    array_type::tensor<size_t, N> m_i; ///< Index of the current yield strain per item.
    array_type::tensor<double, N> m_eps; ///< Equivalent strain.
    array_type::tensor<double, N + 1> m_epsy; ///< Yield strain sequence.
    size_t m_nyield; ///< shape(-1)

    using Elastic<N>::m_K;
    using Elastic<N>::m_G;
    using Elastic<N>::m_Eps;
    using Elastic<N>::m_Sig;
    using Elastic<N>::m_C;

    using GMatTensor::Cartesian2d::Array<N>::m_ndim;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian2d::Array<N>::rank;

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
        this->init_Elastic(K, G);
        m_eps = xt::zeros<double>(m_shape);
        m_i = xt::zeros<size_t>(m_shape);
        this->set_epsy(epsy);
    }

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

    void refresh() override
    {
        namespace GT = GMatTensor::Cartesian2d::pointer;

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

            Sig[0] = Sig[3] = K * epsm;

            if (epsd <= 0.0) {
                Sig[1] = Sig[2] = 0.0;
                continue;
            }

            double eps_min = 0.5 * (*(y + idx) + *(y + idx + 1));

            double g = G * (1.0 - eps_min / epsd);
            Sig[0] += g * Epsd[0];
            Sig[1] = g * Epsd[1];
            Sig[2] = g * Epsd[2];
            Sig[3] += g * Epsd[3];
        }
    }

    array_type::tensor<double, N> energy() const override
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);
        namespace GT = GMatTensor::Cartesian2d::pointer;

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

            double U = K * std::pow(epsm, 2.0);
            double V = G * (std::pow(epsd - eps_min, 2.0) - std::pow(deps_y, 2.0));

            ret.flat(i) = U + V;
        }

        return ret;
    }
};

/**
Array of material points with an elasto-plastic material model.
The response is defined by a potential energy landscape consisting of a sequence of smooth wells.
\tparam N Rank of the array.
*/
template <size_t N>
class Smooth : public Cusp<N> {
protected:
    using Cusp<N>::m_i;
    using Cusp<N>::m_eps;
    using Cusp<N>::m_epsy;
    using Cusp<N>::m_nyield;

    using Elastic<N>::m_K;
    using Elastic<N>::m_G;
    using Elastic<N>::m_Eps;
    using Elastic<N>::m_Sig;
    using Elastic<N>::m_C;

    using GMatTensor::Cartesian2d::Array<N>::m_ndim;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian2d::Array<N>::rank;

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
        this->init_Elastic(K, G);
        m_eps = xt::zeros<double>(m_shape);
        m_i = xt::zeros<size_t>(m_shape);
        this->set_epsy(epsy);
    }

    void refresh() override
    {
        namespace GT = GMatTensor::Cartesian2d::pointer;

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

            Sig[0] = Sig[3] = K * epsm;

            if (epsd <= 0.0) {
                Sig[1] = Sig[2] = 0.0;
                continue;
            }

            double eps_min = 0.5 * (*(y + idx) + *(y + idx + 1));
            double deps_y = 0.5 * (*(y + idx + 1) - *(y + idx));

            double g = (G / epsd) * (deps_y / M_PI) * sin(M_PI / deps_y * (epsd - eps_min));
            Sig[0] += g * Epsd[0];
            Sig[1] = g * Epsd[1];
            Sig[2] = g * Epsd[2];
            Sig[3] += g * Epsd[3];
        }
    }

    array_type::tensor<double, N> energy() const override
    {
        array_type::tensor<double, N> ret = xt::empty<double>(m_shape);
        namespace GT = GMatTensor::Cartesian2d::pointer;

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

            double U = K * std::pow(epsm, 2.0);
            double V = -2.0 * G * std::pow(deps_y / M_PI, 2.0) *
                       (1.0 + cos(M_PI / deps_y * (epsd - eps_min)));

            ret.flat(i) = U + V;
        }

        return ret;
    }
};

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
