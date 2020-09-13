/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

namespace detail
{

    namespace xtensor
    {

        template <class E>
        inline std::vector<size_t> shape(E&& e)
        {
            return std::vector<size_t>(e.shape().cbegin(), e.shape().cend());
        }

        template <class T>
        inline auto trace(const T& A)
        {
            return A(0,0) + A(1,1);
        }

        template <class T, class U>
        inline auto A2_ddot_B2(const T& A, const U& B)
        {
            return A(0,0) * B(0,0) + 2.0 * A(0,1) * B(0,1) + A(1,1) * B(1,1);
        }

    } // namespace xtensor

    namespace pointer
    {

        template <class T>
        inline auto trace(const T A)
        {
            return A[0] + A[3];
        }

        template <class T, class U>
        inline void deviatoric(const T A, U ret)
        {
            auto m = 0.5 * (A[0] + A[3]);
            ret[0] = A[0] - m;
            ret[1] = A[1];
            ret[2] = A[2];
            ret[3] = A[3] - m;
        }

        template <class T>
        inline auto deviatoric_ddot_deviatoric(const T A)
        {
            auto m = 0.5 * (A[0] + A[3]);
            return (A[0] - m) * (A[0] - m) + 2.0 * A[1] * A[1] + (A[3] - m) * (A[3] - m);
        }

    } // namespace pointer

    template <class T>
    inline T trace(const std::array<T,4>& A)
    {
        return A[0] + A[3];
    }

    template <class T>
    inline T hydrostatic_deviator(const std::array<T,4>& A, std::array<T,4>& Ad)
    {
        T Am = 0.5 * (A[0] + A[3]);
        Ad[0] = A[0] - Am;
        Ad[1] = A[1];
        Ad[2] = A[2];
        Ad[3] = A[3] - Am;
        return Am;
    }

    template <class T>
    inline T A2_ddot_B2(const std::array<T,4>& A, const std::array<T,4>& B)
    {
        return A[0] * B[0] + 2.0 * A[1] * B[1] + A[3] * B[3];
    }

} // namespace detail

inline Tensor2 I2()
{
    return Tensor2({{1.0, 0.0},
                    {0.0, 1.0}});
}

inline Tensor4 II()
{
    Tensor4 ret = Tensor4::from_shape({2, 2, 2, 2});
    ret.fill(0.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == j && k == l) {
                        ret(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return ret;
}

inline Tensor4 I4()
{
    Tensor4 ret = Tensor4::from_shape({2, 2, 2, 2});
    ret.fill(0.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == l && j == k) {
                        ret(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return ret;
}

inline Tensor4 I4rt()
{
    Tensor4 ret = Tensor4::from_shape({2, 2, 2, 2});
    ret.fill(0.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == k && j == l) {
                        ret(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return ret;
}

inline Tensor4 I4s()
{
    return 0.5 * (I4() + I4rt());
}

inline Tensor4 I4d()
{
    return I4s() - 0.5 * II();
}

namespace detail
{

    template <class T, typename = void>
    struct equiv_impl
    {
    };

    // rank 2 tensor

    template <class T>
    struct equiv_impl<T, typename std::enable_if_t<xt::has_rank_t<T, 2>::value>>
    {
        using value_type = typename T::value_type;

        static auto deviatoric_alloc(const T& A)
        {
            return A - 0.5 * detail::xtensor::trace(A) * I2();
        }

        static auto hydrostatic_alloc(const T& A)
        {
            return 0.5 * detail::xtensor::trace(A);
        }

        static auto epsd_alloc(const T& A)
        {
            T Ad = A - 0.5 * detail::xtensor::trace(A) * I2();
            return std::sqrt(0.5 * detail::xtensor::A2_ddot_B2(Ad, Ad));
        }

        static auto sigd_alloc(const T& A)
        {
            T Ad = A - 0.5 * detail::xtensor::trace(A) * I2();
            return std::sqrt(2.0 * detail::xtensor::A2_ddot_B2(Ad, Ad));
        }
    };

    // list of rank 2 tensors

    template <class T>
    struct equiv_impl<T, typename std::enable_if_t<xt::has_rank_t<T, 3>::value>>
    {
        using value_type = typename T::value_type;
        using shape_type = typename T::shape_type;

        static void deviatoric_no_alloc(const T& A, xt::xtensor<value_type,3>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == B.shape());
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                detail::pointer::deviatoric(&A(e, 0, 0), &B(e, 0, 0));
            }
        }

        static void hydrostatic_no_alloc(const T& A, xt::xtensor<value_type,1>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == shape_type({B.shape(0), 2, 2}));
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                B(e) = 0.5 * detail::pointer::trace(&A(e, 0, 0));
            }
        }

        static void epsd_no_alloc(const T& A, xt::xtensor<value_type,1>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == shape_type({B.shape(0), 2, 2}));
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                auto b = detail::pointer::deviatoric_ddot_deviatoric(&A(e, 0, 0));
                B(e) = std::sqrt(0.5 * b);
            }
        }

        static void sigd_no_alloc(const T& A, xt::xtensor<value_type,1>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == shape_type({B.shape(0), 2, 2}));
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                auto b = detail::pointer::deviatoric_ddot_deviatoric(&A(e, 0, 0));
                B(e) = std::sqrt(2.0 * b);
            }
        }

        static auto deviatoric_alloc(const T& A)
        {
            xt::xtensor<value_type,3> B = xt::empty<value_type>(A.shape());
            deviatoric_no_alloc(A, B);
            return B;
        }

        static auto hydrostatic_alloc(const T& A)
        {
            xt::xtensor<value_type,1> B = xt::empty<value_type>({A.shape(0)});
            hydrostatic_no_alloc(A, B);
            return B;
        }

        static auto epsd_alloc(const T& A)
        {
            xt::xtensor<value_type,1> B = xt::empty<value_type>({A.shape(0)});
            epsd_no_alloc(A, B);
            return B;
        }

        static auto sigd_alloc(const T& A)
        {
            xt::xtensor<value_type,1> B = xt::empty<value_type>({A.shape(0)});
            sigd_no_alloc(A, B);
            return B;
        }
    };

    // matrix of rank 2 tensors

    template <class T>
    struct equiv_impl<T, typename std::enable_if_t<xt::has_rank_t<T, 4>::value>>
    {
        using value_type = typename T::value_type;
        using shape_type = typename T::shape_type;

        static void deviatoric_no_alloc(const T& A, xt::xtensor<value_type,4>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == B.shape());
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                for (size_t q = 0; q < A.shape(1); ++q) {
                    detail::pointer::deviatoric(&A(e, q, 0, 0), &B(e, q, 0, 0));
                }
            }
        }

        static auto hydrostatic_no_alloc(const T& A, xt::xtensor<value_type,2>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == shape_type({B.shape(0), B.shape(1), 2, 2}));
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                for (size_t q = 0; q < A.shape(1); ++q) {
                    B(e, q) = 0.5 * detail::pointer::trace(&A(e, q, 0, 0));
                }
            }
        }

        static void epsd_no_alloc(const T& A, xt::xtensor<value_type,2>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == shape_type({B.shape(0), B.shape(1), 2, 2}));
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                for (size_t q = 0; q < A.shape(1); ++q) {
                    auto b = detail::pointer::deviatoric_ddot_deviatoric(&A(e, q, 0, 0));
                    B(e, q) = std::sqrt(0.5 * b);
                }
            }
        }

        static void sigd_no_alloc(const T& A, xt::xtensor<value_type,2>& B)
        {
            GMATELASTOPLASTICQPOT_ASSERT(A.shape() == shape_type({B.shape(0), B.shape(1), 2, 2}));
            #pragma omp parallel for
            for (size_t e = 0; e < A.shape(0); ++e) {
                for (size_t q = 0; q < A.shape(1); ++q) {
                    auto b = detail::pointer::deviatoric_ddot_deviatoric(&A(e, q, 0, 0));
                    B(e, q) = std::sqrt(2.0 * b);
                }
            }
        }

        static auto deviatoric_alloc(const T& A)
        {
            xt::xtensor<value_type,4> B = xt::empty<value_type>(A.shape());
            deviatoric_no_alloc(A, B);
            return B;
        }

        static auto hydrostatic_alloc(const T& A)
        {
            xt::xtensor<value_type,2> B = xt::empty<value_type>({A.shape(0), A.shape(1)});
            hydrostatic_no_alloc(A, B);
            return B;
        }

        static auto epsd_alloc(const T& A)
        {
            xt::xtensor<value_type,2> B = xt::empty<value_type>({A.shape(0), A.shape(1)});
            epsd_no_alloc(A, B);
            return B;
        }

        static auto sigd_alloc(const T& A)
        {
            xt::xtensor<value_type,2> B = xt::empty<value_type>({A.shape(0), A.shape(1)});
            sigd_no_alloc(A, B);
            return B;
        }
    };

} // namespace detail

template <class T, class U>
inline void hydrostatic(const T& A, U& B)
{
    return detail::equiv_impl<T>::hydrostatic_no_alloc(A, B);
}

template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::equiv_impl<T>::hydrostatic_alloc(A);
}

template <class T, class U>
inline void deviatoric(const T& A, U& B)
{
    return detail::equiv_impl<T>::deviatoric_no_alloc(A, B);
}

template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::equiv_impl<T>::deviatoric_alloc(A);
}

template <class T, class U>
inline void epsd(const T& A, U& B)
{
    return detail::equiv_impl<T>::epsd_no_alloc(A, B);
}

template <class T>
inline auto Epsd(const T& A)
{
    return detail::equiv_impl<T>::epsd_alloc(A);
}

template <class T, class U>
inline void sigd(const T& A, U& B)
{
    return detail::equiv_impl<T>::sigd_no_alloc(A, B);
}

template <class T>
inline auto Sigd(const T& A)
{
    return detail::equiv_impl<T>::sigd_alloc(A);
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
