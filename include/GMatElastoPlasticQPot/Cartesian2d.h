/**
Implementation in a 2-d Cartesian coordinate frame.

\file
\copyright Copyright 2018. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_H
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_H

// use "M_PI" from "math.h"
#define _USE_MATH_DEFINES

#include <QPot/Chunked.hpp>
#include <GMatTensor/Cartesian2d.h>
#include <math.h>
#include <xtensor/xsort.hpp>

#include "config.h"

namespace GMatElastoPlasticQPot {

/**
Implementation in a 2-d Cartesian coordinate frame.

Note that for convenience this namespace include aliases to:
-   GMatTensor::Cartesian2d::O2()
-   GMatTensor::Cartesian2d::O4()
-   GMatTensor::Cartesian2d::I2()
-   GMatTensor::Cartesian2d::II()
-   GMatTensor::Cartesian2d::I4()
-   GMatTensor::Cartesian2d::I4rt()
-   GMatTensor::Cartesian2d::I4s()
-   GMatTensor::Cartesian2d::I4d()
-   GMatTensor::Cartesian2d::Hydrostatic()
-   GMatTensor::Cartesian2d::hydrostatic()
-   GMatTensor::Cartesian2d::Deviatoric()
-   GMatTensor::Cartesian2d::deviatoric()
*/
namespace Cartesian2d {

using GMatTensor::Cartesian2d::O2;
using GMatTensor::Cartesian2d::O4;
using GMatTensor::Cartesian2d::I2;
using GMatTensor::Cartesian2d::II;
using GMatTensor::Cartesian2d::I4;
using GMatTensor::Cartesian2d::I4rt;
using GMatTensor::Cartesian2d::I4s;
using GMatTensor::Cartesian2d::I4d;
using GMatTensor::Cartesian2d::Hydrostatic;
using GMatTensor::Cartesian2d::hydrostatic;
using GMatTensor::Cartesian2d::Deviatoric;
using GMatTensor::Cartesian2d::deviatoric;

/**
Equivalent strain: norm of strain deviator

\f$ \sqrt{\frac{1}{2} (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use epsd().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Epsd(const T& A);

/**
Same as Epsd(), but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array
*/
template <class T, class U>
inline void epsd(const T& A, U& ret);

/**
Equivalent strain: norm of strain deviator

\f$ \sqrt{2 (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use sigd().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Sigd(const T& A);

/**
Same as Sigd(), but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array
*/
template <class T, class U>
inline void sigd(const T& A, U& ret);

/**
Elastic material point.
*/
class Elastic
{
public:

    Elastic() = default;

    /**
    Constructor.

    \param K Bulk modulus.
    \param G Shear modulus.
    */
    Elastic(double K, double G);

    /**
    \return Bulk modulus.
    */
    double K() const;

    /**
    \return Shear modulus.
    */
    double G() const;

    /**
    \return Current potential energy.
    */
    double energy() const;

    /**
    Set the current strain tensor.

    \param arg xtensor array [2, 2].
    */
    template <class T>
    void setStrain(const T& arg);

    /**
    Same as setStrain(), but reads from a pointer assuming row-major storage (no bound check).

    \param arg Pointer to array (xx, xy, yx, yy).
    */
    template <class T>
    void setStrainPtr(const T* arg);

    /**
    Get the current strain tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Strain() const;

    /**
    Same as Strain(), but write to allocated data.

    \param ret xtensor array [2, 2], overwritten.
    */
    template <class T>
    void strain(T& ret) const;

    /**
    Same as Strain(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void strainPtr(T* ret) const;

    /**
    Get the current stress tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Stress() const;

    /**
    Same as Stress(), but write to allocated data.

    \param ret xtensor array [2, 2], overwritten.
    */
    template <class T>
    void stress(T& ret) const;

    /**
    Same as Stress(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void stressPtr(T* ret) const;

    /**
    Get the tangent tensor (strain independent).

    \return [2, 2, 2, 2] array.
    */
    xt::xtensor<double, 4> Tangent() const;

    /**
    Same as Tangent(), but write to allocated data.

    \param ret xtensor array [2, 2, 2, 2], overwritten.
    */
    template <class T>
    void tangent(T& ret) const;

    /**
    Same as Tangent(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array of size 2 * 2 * 2 * 2, overwritten.
    */
    template <class T>
    void tangentPtr(T* ret) const;

private:
    double m_K; ///< bulk modulus
    double m_G; ///< shear modulus
    std::array<double, 4> m_Eps; ///< strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; ///< stress tensor ,,
};

/**
Elasto-plastic material point.
Defined by a potential energy landscape consisting of a sequence of parabolic potentials.
*/
class Cusp
{
public:

    Cusp() = default;

    /**
    Constructor.

    \param K Bulk modulus.
    \param G Shear modulus.
    \param epsy Sequence of yield strains.
    \param init_elastic Initialise in minimum at zero strain
        If set `true`, `epsy` is prepended with  `- epsy(0)`.
        In that case, one should remember that the first yield strain specified as `epsy`,
        will be the second yield strain held in storage. I.e the storage will be
        as follows: `[- epsy(0), epsy(0), epsy(1), epsy(2), ...].
        It is crucial to pay attention to this when using chunked storage.
    */
    template <class Y>
    Cusp(double K, double G, const Y&, bool init_elastic = true);

    /**
    \return Bulk modulus.
    */
    double K() const;

    /**
    \return Shear modulus.
    */
    double G() const;

    /**
    \return Yield strains.
    */
    xt::xtensor<double, 1> epsy() const;

    /**
    \return Reference to the underlying #QPot::Chunked model.
    */
    QPot::Chunked& refQPotChunked();

    /**
    \return Current yield index, see QPot::Chunked::i().
    */
    auto currentIndex() const;

    /**
    \return Current yield strain left, see QPot::Chunked::yleft().
    */
    auto currentYieldLeft() const;

    /**
    \return Current yield strain right, see QPot::Chunked::yright().
    */
    auto currentYieldRight() const;

    /**
    \return Current yield strain at an offset left,
    see QPot::Chunked::yleft(size_t) const
    */
    auto currentYieldLeft(size_t offset) const;

    /**
    \return Current yield strain at an offset right,
    see QPot::Chunked::yright(size_t) const
    */
    auto currentYieldRight(size_t offset) const;

    /**
    \return Plastic strain = 0.5 * (currentYieldLeft() + currentYieldRight())
    */
    double epsp() const;

    /**
    \return Current potential energy.
    */
    double energy() const;

    /**
    \return See QPot::Chunked::checkYieldBoundLeft()
    */
    bool checkYieldBoundLeft(size_t n = 0) const;

    /**
    \return See QPot::Chunked::checkYieldBoundRight()
    */
    bool checkYieldBoundRight(size_t n = 0) const;

    /**
    Set the current strain tensor.

    \param arg xtensor array [2, 2].
    */
    template <class T>
    void setStrain(const T& arg);

    /**
    Same as setStrain(), but reads from a pointer assuming row-major storage (no bound check).

    \param arg Pointer to array (xx, xy, yx, yy).
    */
    template <class T>
    void setStrainPtr(const T* arg);

    /**
    Get the current strain tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Strain() const;

    /**
    Same as Strain(), but write to allocated data.

    \param ret xtensor array [2, 2], overwritten.
    */
    template <class T>
    void strain(T& ret) const;

    /**
    Same as Strain(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void strainPtr(T* ret) const;

    /**
    Get the current stress tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Stress() const;

    /**
    Same as Stress(), but write to allocated data.

    \param ret xtensor array [2, 2], overwritten.
    */
    template <class T>
    void stress(T& ret) const;

    /**
    Same as Stress(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void stressPtr(T* ret) const;

    /**
    Get the tangent tensor (strain independent).

    \return [2, 2, 2, 2] array.
    */
    xt::xtensor<double, 4> Tangent() const;

    /**
    Same as Tangent(), but write to allocated data.

    \param ret xtensor array [2, 2, 2, 2], overwritten.
    */
    template <class T>
    void tangent(T& ret) const;

    /**
    Same as Tangent(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array of size 2 * 2 * 2 * 2, overwritten.
    */
    template <class T>
    void tangentPtr(T* ret) const;

private:
    double m_K; ///< bulk modulus
    double m_G; ///< shear modulus
    QPot::Chunked m_yield; ///< potential energy landscape
    std::array<double, 4> m_Eps; ///< strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; ///< stress tensor ,,
};

/**
Elasto-plastic material point.
Defined by a potential energy landscape consisting of a sequence of smoothed parabolic potentials.
*/
class Smooth
{
public:

    Smooth() = default;

    /**
    Constructor.

    \param K Bulk modulus.
    \param G Shear modulus.
    \param epsy Sequence of yield strains.
    \param init_elastic Initialise in minimum at zero strain
        If set `true`, `epsy` is prepended with  `- epsy(0)`.
        In that case, one should remember that the first yield strain specified as `epsy`,
        will be the second yield strain held in storage. I.e the storage will be
        as follows: `[- epsy(0), epsy(0), epsy(1), epsy(2), ...].
        It is crucial to pay attention to this when using chunked storage.
    */
    template <class Y>
    Smooth(double K, double G, const Y& epsy, bool init_elastic = true);

    /**
    \return Bulk modulus.
    */
    double K() const;

    /**
    \return Shear modulus.
    */
    double G() const;

    /**
    \return Yield strains.
    */
    xt::xtensor<double, 1> epsy() const;

    /**
    \return Reference to the underlying #QPot::Chunked model.
    */
    QPot::Chunked& refQPotChunked();

    /**
    \return Current yield index, see QPot::Chunked::i().
    */
    auto currentIndex() const;

    /**
    \return Current yield strain left, see QPot::Chunked::yleft().
    */
    auto currentYieldLeft() const;

    /**
    \return Current yield strain right, see QPot::Chunked::yright().
    */
    auto currentYieldRight() const;

    /**
    \return Current yield strain at an offset left,
    see QPot::Chunked::yleft(size_t) const
    */
    auto currentYieldLeft(size_t offset) const;

    /**
    \return Current yield strain at an offset right,
    see QPot::Chunked::yright(size_t) const
    */
    auto currentYieldRight(size_t offset) const;

    /**
    \return Plastic strain = 0.5 * (currentYieldLeft() + currentYieldRight())
    */
    double epsp() const;

    /**
    \return Current potential energy.
    */
    double energy() const;

    /**
    \return See QPot::Chunked::checkYieldBoundLeft()
    */
    bool checkYieldBoundLeft(size_t n = 0) const;

    /**
    \return See QPot::Chunked::checkYieldBoundRight()
    */
    bool checkYieldBoundRight(size_t n = 0) const;

    /**
    Set the current strain tensor.

    \param arg xtensor array [2, 2].
    */
    template <class T>
    void setStrain(const T& arg);

    /**
    Same as setStrain(), but reads from a pointer assuming row-major storage (no bound check).

    \param arg Pointer to array (xx, xy, yx, yy).
    */
    template <class T>
    void setStrainPtr(const T* arg);

    /**
    Get the current strain tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Strain() const;

    /**
    Same as Strain(), but write to allocated data.

    \param ret xtensor array [2, 2], overwritten.
    */
    template <class T>
    void strain(T& ret) const;

    /**
    Same as Strain(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void strainPtr(T* ret) const;

    /**
    Get the current stress tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Stress() const;

    /**
    Same as Stress(), but write to allocated data.

    \param ret xtensor array [2, 2], overwritten.
    */
    template <class T>
    void stress(T& ret) const;

    /**
    Same as Stress(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void stressPtr(T* ret) const;

    /**
    Get the tangent tensor (strain independent).

    \return [2, 2, 2, 2] array.
    */
    xt::xtensor<double, 4> Tangent() const;

    /**
    Same as Tangent(), but write to allocated data.

    \param ret xtensor array [2, 2, 2, 2], overwritten.
    */
    template <class T>
    void tangent(T& ret) const;

    /**
    Same as Tangent(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array of size 2 * 2 * 2 * 2, overwritten.
    */
    template <class T>
    void tangentPtr(T* ret) const;

private:
    double m_K; ///< bulk modulus
    double m_G; ///< shear modulus
    QPot::Chunked m_yield; ///< potential energy landscape
    std::array<double, 4> m_Eps; ///< strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; ///< stress tensor ,,
};

/**
Material identifier.
*/
struct Type {
    /**
    Type value.
    */
    enum Value {
        Unset,   ///< Unset
        Elastic, ///< See Elastic
        Cusp,    ///< See Cusp
        Smooth,  ///< See Smooth
    };
};

/**
Array of material points.

\tparam N Rank of the array.
*/
template <size_t N>
class Array : public GMatTensor::Cartesian2d::Array<N>
{
public:

    using GMatTensor::Cartesian2d::Array<N>::rank;

    Array() = default;

    /**
    Basic constructor.
    Note that before usage material properties still have to be assigned to all items.
    This can be done per item or by groups of items, using:
    -   setElastic()
    -   setCusp()
    -   setSmooth()

    \param shape The shape of the array.
    */
    Array(const std::array<size_t, N>& shape);

    /**
    \return Type-id per item. Follows order set in Type.
    */
    xt::xtensor<size_t, N> type() const;

    /**
    \return Per item, 1 if Elastic, otherwise 0.
    */
    xt::xtensor<bool, N> isElastic() const;

    /**
    \return Per item, 1 if Cusp or Smooth, otherwise 0.
    */
    xt::xtensor<bool, N> isPlastic() const;

    /**
    \return Per item, 1 if Cusp, otherwise 0.
    */
    xt::xtensor<bool, N> isCusp() const;

    /**
    \return Per item, 1 if Smooth, otherwise 0.
    */
    xt::xtensor<bool, N> isSmooth() const;

    /**
    \return Bulk modulus per item.
    */
    xt::xtensor<double, N> K() const;

    /**
    \return Shear modulus per item.
    */
    xt::xtensor<double, N> G() const;

    /**
    Set all items Elastic, specifying material parameters per item.

    \param K Bulk modulus per item [shape()].
    \param G Shear modulus per item [shape()].
    */
    void setElastic(
        const xt::xtensor<double, N>& K,
        const xt::xtensor<double, N>& G);

    /**
    Set a batch of items Elastic, with the material parameters the same for all set items.

    \param I Per item, ``true`` to set Elastic, ``false`` to skip.
    \param K Bulk modulus.
    \param G Shear modulus.
    */
    void setElastic(
        const xt::xtensor<bool, N>& I,
        double K,
        double G);

    /**
    Set a batch of items Cusp, with the material parameters the same for all set items.

    \param I Per item, ``true`` to set Cusp, ``false`` to skip.
    \param K Bulk modulus.
    \param G Shear modulus.
    \param epsy Sequence of yield strains.
    \param init_elastic Initialise in minimum at zero strain.
    */
    void setCusp(
        const xt::xtensor<bool, N>& I,
        double K,
        double G,
        const xt::xtensor<double, 1>& epsy,
        bool init_elastic = true);

    /**
    Set a batch of items Smooth, with the material parameters the same for all set items.

    \param I Per item, ``true`` to set Smooth, ``false`` to skip.
    \param K Bulk modulus.
    \param G Shear modulus.
    \param epsy Sequence of yield strains.
    \param init_elastic Initialise in minimum at zero strain.
    */
    void setSmooth(
        const xt::xtensor<bool, N>& I,
        double K,
        double G,
        const xt::xtensor<double, 1>& epsy,
        bool init_elastic = true);

    /**
    Set a batch of items Elastic, with the material parameters (possibly) different.
    To this end, and addition array ``idx`` is used that refers to a which entry to use:
    ``K(idx)``, ``G(idx)``, and ``epsy(idx, :)``.

    \param I Per item, ``true`` to set Elastic, ``false`` to skip.
    \param idx Per item, index in supplied material parameters.
    \param K Bulk modulus.
    \param G Shear modulus.
    */
    void setElastic(
        const xt::xtensor<bool, N>& I,
        const xt::xtensor<size_t, N>& idx,
        const xt::xtensor<double, 1>& K,
        const xt::xtensor<double, 1>& G);

    /**
    Set a batch of items Cusp, with the material parameters (possibly) different.
    To this end, and addition array ``idx`` is used that refers to a which entry to use:
    ``K(idx)``, ``G(idx)``, and ``epsy(idx, :)``.

    \param I Per item, ``true`` to set Cusp, ``false`` to skip.
    \param idx Per item, index in supplied material parameters.
    \param K Bulk modulus.
    \param G Shear modulus.
    \param epsy Sequence of yield strains.
    \param init_elastic Initialise in minimum at zero strain.
    */
    void setCusp(
        const xt::xtensor<bool, N>& I,
        const xt::xtensor<size_t, N>& idx,
        const xt::xtensor<double, 1>& K,
        const xt::xtensor<double, 1>& G,
        const xt::xtensor<double, 2>& epsy,
        bool init_elastic = true);

    /**
    Set a batch of items Smooth, with the material parameters (possibly) different.
    To this end, and addition array ``idx`` is used that refers to a which entry to use:
    ``K(idx)``, ``G(idx)``, and ``epsy(idx, :)``.

    \param I Per item, ``true`` to set Smooth, ``false`` to skip.
    \param idx Per item, index in supplied material parameters.
    \param K Bulk modulus.
    \param G Shear modulus.
    \param epsy Sequence of yield strains.
    \param init_elastic Initialise in minimum at zero strain.
    */
    void setSmooth(
        const xt::xtensor<bool, N>& I,
        const xt::xtensor<size_t, N>& idx,
        const xt::xtensor<double, 1>& K,
        const xt::xtensor<double, 1>& G,
        const xt::xtensor<double, 2>& epsy,
        bool init_elastic = true);

    /**
    Set strain tensors.

    \param arg Strain tensor per item [shape(), 2, 2].
    */
    void setStrain(const xt::xtensor<double, N + 2>& arg);

    /**
    \return Strain tensor per item [shape(), 2, 2].
    */
    xt::xtensor<double, N + 2> Strain() const;

    /**
    Same as Strain(), but write to allocated data.

    \param ret [shape(), 2, 2], overwritten.
    */
    void strain(xt::xtensor<double, N + 2>& ret) const;

    /**
    \return Stress tensor per item [shape(), 2, 2].
    */
    xt::xtensor<double, N + 2> Stress() const;

    /**
    Same as Stress(), but write to allocated data.

    \param ret [shape(), 2, 2], overwritten.
    */
    void stress(xt::xtensor<double, N + 2>& ret) const;

    /**
    \return Tangent tensor per item [shape(), 2, 2].
    */
    xt::xtensor<double, N + 4> Tangent() const;

    /**
    Same as Tangent(), but write to allocated data.

    \param ret [shape(), 2, 2, 2, 2], overwritten.
    */
    void tangent(xt::xtensor<double, N + 4>& ret) const;

    /**
    \return Yield index per item [shape()], see QPot::Chunked::i().
    */
    xt::xtensor<long, N> CurrentIndex() const;

    /**
    Same as CurrentIndex(), but write to allocated data.

    \param ret [shape()], overwritten.
    */
    void currentIndex(xt::xtensor<long, N>& ret) const;

    /**
    \return Yield strain left [shape()], see QPot::Chunked::yleft().
    */
    xt::xtensor<double, N> CurrentYieldLeft() const;

    /**
    Same as CurrentYieldLeft(), but write to allocated data.

    \param ret [shape()], overwritten.
    */
    void currentYieldLeft(xt::xtensor<double, N>& ret) const;

    /**
    \return Yield strain right [shape()], see QPot::Chunked::yright().
    */
    xt::xtensor<double, N> CurrentYieldRight() const;

    /**
    Same as CurrentYieldRight(), but write to allocated data.

    \param ret [shape()], overwritten.
    */
    void currentYieldRight(xt::xtensor<double, N>& ret) const;

    /**
    \return Yield strain at an offset left [shape()], see QPot::Chunked::yleft().
    */
    xt::xtensor<double, N> CurrentYieldLeft(size_t offset) const;

    /**
    Same as CurrentYieldLeft(), but write to allocated data.

    \param ret [shape()], overwritten.
    \param offset
    */
    void currentYieldLeft(xt::xtensor<double, N>& ret, size_t offset) const;

    /**
    \param offset
    \return Yield strain at an offset right [shape()], see QPot::Chunked::yright().
    */
    xt::xtensor<double, N> CurrentYieldRight(size_t offset) const;

    /**
    Same as CurrentYieldRight(), but write to allocated data.

    \param ret [shape()], overwritten.
    \param offset
    */
    void currentYieldRight(xt::xtensor<double, N>& ret, size_t offset) const;

    /**
    \param n Number of potentials that should be remaining to the left.
    \return Bound check, see QPot::Chunked::checkYieldBoundLeft().
    */
    bool checkYieldBoundLeft(size_t n = 0) const;

    /**
    \param n Number of potentials that should be remaining to the right.
    \return Bound check, see QPot::Chunked::checkYieldBoundRight().
    */
    bool checkYieldBoundRight(size_t n = 0) const;

    /**
    \return Plastic strain [shape()].
    */
    xt::xtensor<double, N> Epsp() const;

    /**
    Same as Epsp(), but write to allocated data.

    \param ret [shape()], overwritten.
    */
    void epsp(xt::xtensor<double, N>& ret) const;

    /**
    \return Elastic energy item [shape()].
    */
    xt::xtensor<double, N> Energy() const;

    /**
    Same as Energy(), but write to allocated data.

    \param ret [shape()], overwritten.
    */
    void energy(xt::xtensor<double, N>& ret) const;

    /**
    Copy to the underlying Elastic model of an item.

    \param index The index of the item.
    \return Copy to the model.
    */
    [[ deprecated ]]
    auto getElastic(const std::array<size_t, N>& index) const;

    /**
    Copy to the underlying Cusp model of an item.

    \param index The index of the item.
    \return Copy to the model.
    */
    [[ deprecated ]]
    auto getCusp(const std::array<size_t, N>& index) const;

    /**
    Copy to the underlying Smooth model of an item.

    \param index The index of the item.
    \return Copy to the model.
    */
    [[ deprecated ]]
    auto getSmooth(const std::array<size_t, N>& index) const;

    /**
    Reference to the underlying Elastic model of an item.

    \param index The index of the item.
    \return Reference to the model.
    */
    Elastic& refElastic(const std::array<size_t, N>& index);

    /**
    Reference to the underlying Cusp model of an item.

    \param index The index of the item.
    \return Reference to the model.
    */
    Cusp& refCusp(const std::array<size_t, N>& index);

    /**
    Reference to the underlying Smooth model of an item.

    \param index The index of the item.
    \return Reference to the model.
    */
    Smooth& refSmooth(const std::array<size_t, N>& index);

private:

    /**
    Elastic material vectors: each item has one entry in one of the material vectors.
    */
    std::vector<Elastic> m_Elastic;

    /**
    Cusp material vectors: each item has one entry in one of the material vectors.
    */
    std::vector<Cusp> m_Cusp;

    /**
    Smooth material vectors: each item has one entry in one of the material vectors.
    */
    std::vector<Smooth> m_Smooth;

    /**
    Type of each entry, see Type.
    */
    xt::xtensor<size_t, N> m_type;

    /**
    Index in the relevant material vector (#m_Elastic, #m_Cusp, #m_Smooth)
    */
    xt::xtensor<size_t, N> m_index;

    using GMatTensor::Cartesian2d::Array<N>::m_ndim;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;
};

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#include "Cartesian2d.hpp"
#include "Cartesian2d_Array.hpp"
#include "Cartesian2d_Cusp.hpp"
#include "Cartesian2d_Elastic.hpp"
#include "Cartesian2d_Smooth.hpp"

#endif
