/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>

// Enable basic assertions on matrix shape
// (doesn't cost a lot of time, but avoids segmentation faults)
#define GMATELASTOPLASTICQPOT_ENABLE_ASSERT

#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace py = pybind11;

PYBIND11_MODULE(GMatElastoPlasticQPot, m)
{

m.doc() = "Elasto-plastic material model";

// ---------------------------------
// GMatElastoPlasticQPot.Cartesian2d
// ---------------------------------

py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");

namespace SM = GMatElastoPlasticQPot::Cartesian2d;

// Unit tensors

sm.def("I2", &SM::I2, "Second order unit tensor.");
sm.def("II", &SM::II, "Fourth order tensor with the result of the dyadic product II.");
sm.def("I4", &SM::I4, "Fourth order unit tensor.");
sm.def("I4rt", &SM::I4rt, "Fourth right-transposed order unit tensor.");
sm.def("I4s", &SM::I4s, "Fourth order symmetric projection tensor.");
sm.def("I4d", &SM::I4d, "Fourth order deviatoric projection tensor.");

// Tensor algebra

sm.def("Deviatoric",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Deviatoric<xt::xtensor<double,4>>),
    "Deviatoric part of a 2nd-order tensor. Returns matrix of 2nd-order tensors.",
    py::arg("A"));

sm.def("Deviatoric",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Deviatoric<xt::xtensor<double,3>>),
    "Deviatoric part of a 2nd-order tensor. Returns list of 2nd-order tensors.",
    py::arg("A"));

sm.def("Deviatoric",
    py::overload_cast<const SM::Tensor2&>(&SM::Deviatoric<SM::Tensor2>),
    "Deviatoric part of a 2nd-order tensor. Returns 2nd-order tensor.",
    py::arg("A"));

sm.def("Hydrostatic",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Hydrostatic<xt::xtensor<double,4>>),
    "Hydrostatic part of a 2nd-order tensor. Returns matrix (of scalars).",
    py::arg("A"));

sm.def("Hydrostatic",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Hydrostatic<xt::xtensor<double,3>>),
    "Hydrostatic part of a 2nd-order tensor. Returns list (of scalars).",
    py::arg("A"));

sm.def("Hydrostatic",
    py::overload_cast<const SM::Tensor2&>(&SM::Hydrostatic<SM::Tensor2>),
    "Hydrostatic part of a 2nd-order tensor. Returns scalar.",
    py::arg("A"));

sm.def("Epsd",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Epsd<xt::xtensor<double,4>>),
    "Equivalent strain. Returns matrix (of scalars).",
    py::arg("A"));

sm.def("Epsd",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Epsd<xt::xtensor<double,3>>),
    "Equivalent strain. Returns list (of scalars).",
    py::arg("A"));

sm.def("Epsd",
    py::overload_cast<const SM::Tensor2&>(&SM::Epsd<SM::Tensor2>),
    "Equivalent strain. Returns scalar.",
    py::arg("A"));

sm.def("Sigd",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Sigd<xt::xtensor<double,4>>),
    "Equivalent stress. Returns matrix (of scalars).",
    py::arg("A"));

sm.def("Sigd",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Sigd<xt::xtensor<double,3>>),
    "Equivalent stress. Returns list (of scalars).",
    py::arg("A"));

sm.def("Sigd",
    py::overload_cast<const SM::Tensor2&>(&SM::Sigd<SM::Tensor2>),
    "Equivalent stress. Returns scalar.",
    py::arg("A"));



// Material point: Elastic

py::class_<SM::Elastic>(sm, "Elastic")

    .def(py::init<double, double>(),
        "Elastic material point.",
        py::arg("K"),
        py::arg("G"))

    .def("K", &SM::Elastic::K, "Returns the bulk modulus.")

    .def("G", &SM::Elastic::G, "Returns the shear modulus.")

    .def("setStrain",
        &SM::Elastic::setStrain<SM::Tensor2>,
        "Set current strain tensor.")

    .def("Stress",
        &SM::Elastic::Stress,
        "Returns stress tensor, for last known strain.")

    .def("Tangent",
        &SM::Elastic::Tangent,
        "Returns stress and tangent stiffness tensors, for last known strain.")

    .def("energy",
        &SM::Elastic::energy,
        "Returns the energy, for last known strain.")

    .def("__repr__", [](const SM::Elastic&) {
        return "<GMatElastoPlasticQPot.Cartesian2d.Elastic>";
    });

// Material point: Cusp

py::class_<SM::Cusp>(sm, "Cusp")

    .def(py::init<double, double, const xt::xtensor<double,1>&, bool>(),
        "Elasto-plastic material point, with 'cusp' potentials.",
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"),
        py::arg("init_elastic") = true)

    .def("K", &SM::Cusp::K, "Returns the bulk modulus.")

    .def("G", &SM::Cusp::G, "Returns the shear modulus.")

    .def("epsy",
        &SM::Cusp::epsy,
        "Returns the yield strains.")

    .def("setStrain",
        &SM::Cusp::setStrain<SM::Tensor2>,
        "Set current strain tensor.")

    .def("Stress",
        &SM::Cusp::Stress,
        "Returns stress tensor, for last known strain.")

    .def("Tangent",
        &SM::Cusp::Tangent,
        "Returns stress and tangent stiffness tensors, for last known strain.")

    .def("currentIndex",
        &SM::Cusp::currentIndex,
        "Returns the potential index, for last known strain.")

    .def("currentYieldLeft",
        &SM::Cusp::currentYieldLeft,
        "Returns the yield strain to the left, for last known strain.")

    .def("currentYieldRight",
        &SM::Cusp::currentYieldRight,
        "Returns the yield strain to the right, for last known strain.")

    .def("epsp",
        &SM::Cusp::epsp,
        "Returns the equivalent plastic strain for last known strain.")

    .def("energy",
        &SM::Cusp::energy,
        "Returns the energy, for last known strain.")

    .def("__repr__", [](const SM::Cusp&) {
        return "<GMatElastoPlasticQPot.Cartesian2d.Cusp>";
    });

// Material point: Smooth

py::class_<SM::Smooth>(sm, "Smooth")

    .def(py::init<double, double, const xt::xtensor<double,1>&, bool>(),
        "Elasto-plastic material point, with 'smooth' potentials.",
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"),
        py::arg("init_elastic") = true)

    .def("K", &SM::Smooth::K, "Returns the bulk modulus.")

    .def("G", &SM::Smooth::G, "Returns the shear modulus.")

    .def("epsy",
        &SM::Smooth::epsy,
        "Returns the yield strains.")

    .def("setStrain",
        &SM::Smooth::setStrain<SM::Tensor2>,
        "Set current strain tensor.")

    .def("Stress",
        &SM::Smooth::Stress,
        "Returns stress tensor, for last known strain.")

    .def("Tangent",
        &SM::Smooth::Tangent,
        "Returns stress and tangent stiffness tensors, for last known strain.")

    .def("currentIndex",
        &SM::Smooth::currentIndex,
        "Returns the potential index, for last known strain.")

    .def("currentYieldLeft",
        &SM::Smooth::currentYieldLeft,
        "Returns the yield strain to the left, for last known strain.")

    .def("currentYieldRight",
        &SM::Smooth::currentYieldRight,
        "Returns the yield strain to the right, for last known strain.")

    .def("epsp",
        &SM::Smooth::epsp,
        "Returns the equivalent plastic strain for last known strain.")

    .def("energy",
        &SM::Smooth::energy,
        "Returns the energy, for last known strain.")

    .def("__repr__", [](const SM::Smooth&) {
        return "<GMatElastoPlasticQPot.Cartesian2d.Smooth>";
    });

// Material identifier

py::module smm = sm.def_submodule("Type", "Type enumerator");

py::enum_<SM::Type::Value>(smm, "Type")
    .value("Unset", SM::Type::Unset)
    .value("Elastic", SM::Type::Elastic)
    .value("Cusp", SM::Type::Cusp)
    .value("Smooth", SM::Type::Smooth)
    .export_values();

// Matrix

py::class_<SM::Matrix>(sm, "Matrix")

    .def(py::init<size_t, size_t>(),
        "Matrix of material points.",
        py::arg("nelem"),
        py::arg("nip"))

    .def("ndim", &SM::Matrix::ndim, "Return number of (tensor) dimensions.")

    .def("nelem", &SM::Matrix::nelem, "Return number of elements (matrix rows).")

    .def("nip", &SM::Matrix::nip, "Return number of integration points (matrix columns).")

    .def("K", &SM::Matrix::K, "Return matrix with bulk moduli.")

    .def("G", &SM::Matrix::G, "Return matrix with shear moduli.")

    .def("I2", &SM::Matrix::I2, "Return matrix with second order unit tensors.")

    .def("II",
        &SM::Matrix::II,
        "Return matrix with fourth order tensors with the result of the dyadic product II.")

    .def("I4", &SM::Matrix::I4, "Return matrix with fourth order unit tensors.")

    .def("I4rt",
        &SM::Matrix::I4rt,
        "Return matrix with fourth right-transposed order unit tensors.")

    .def("I4s",
        &SM::Matrix::I4s,
        "Return matrix with fourth order symmetric projection tensors.")

    .def("I4d",
        &SM::Matrix::I4d,
        "Return matrix with fourth order deviatoric projection tensors.")

    .def("type", &SM::Matrix::type, "Return matrix with material types.")

    .def("isElastic",
        &SM::Matrix::isElastic,
        "Return matrix with boolean: Elastic (1) or not (0).")

    .def("isPlastic",
        &SM::Matrix::isPlastic,
        "Return matrix with boolean: Elastic (0) or plastic (Cusp/Smooth) (1).")

    .def("isCusp",
        &SM::Matrix::isCusp,
        "Return matrix with boolean: Cusp (1) or not (0).")

    .def("isSmooth",
        &SM::Matrix::isSmooth,
        "Return matrix with boolean: Smooth (1) or not (0).")

    .def("check",
        &SM::Matrix::check,
        "Check that all matrix entries are set. Throws if any unset point is found.")

    .def("setElastic",
        py::overload_cast<
            const xt::xtensor<size_t,2>&,
            const xt::xtensor<size_t,2>&,
            const xt::xtensor<double,1>&,
            const xt::xtensor<double,1>&>(&SM::Matrix::setElastic),
        "Set specific entries 'Elastic'.",
        py::arg("I"),
        py::arg("idx"),
        py::arg("K"),
        py::arg("G"))

    .def("setCusp",
        py::overload_cast<
            const xt::xtensor<size_t,2>&,
            const xt::xtensor<size_t,2>&,
            const xt::xtensor<double,1>&,
            const xt::xtensor<double,1>&,
            const xt::xtensor<double,2>&,
            bool>(&SM::Matrix::setCusp),
        "Set specific entries 'Cusp'.",
        py::arg("I"),
        py::arg("idx"),
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"),
        py::arg("init_elastic") = true)

    .def("setSmooth",
        py::overload_cast<
            const xt::xtensor<size_t,2>&,
            const xt::xtensor<size_t,2>&,
            const xt::xtensor<double,1>&,
            const xt::xtensor<double,1>&,
            const xt::xtensor<double,2>&,
            bool>(&SM::Matrix::setSmooth),
        "Set specific entries 'Smooth'.",
        py::arg("I"),
        py::arg("idx"),
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"),
        py::arg("init_elastic") = true)

    .def("setElastic",
        py::overload_cast<const xt::xtensor<size_t,2>&, double, double>(
            &SM::Matrix::setElastic),
        "Set specific entries 'Elastic'.",
        py::arg("I"),
        py::arg("K"),
        py::arg("G"))

    .def("setCusp",
        py::overload_cast<
            const xt::xtensor<size_t,2>&,
            double,
            double,
            const xt::xtensor<double,1>&,
            bool>(&SM::Matrix::setCusp),
        "Set specific entries 'Cusp'.",
        py::arg("I"),
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"),
        py::arg("init_elastic") = true)

    .def("setSmooth",
        py::overload_cast<
            const xt::xtensor<size_t,2>&,
            double,
            double,
            const xt::xtensor<double,1>&,
            bool>(&SM::Matrix::setSmooth),
        "Set specific entries 'Smooth'.",
        py::arg("I"),
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"),
        py::arg("init_elastic") = true)

    .def("setStrain",
        &SM::Matrix::setStrain,
        "Set matrix of strain tensors.",
        py::arg("Eps"))

    .def("Stress",
        &SM::Matrix::Stress,
        "Returns matrix of stress tensors, given the current strain.")

    .def("Tangent",
        &SM::Matrix::Tangent,
        "Returns matrices of stress tangent stiffness tensors, given the current strain.")

    .def("currentIndex",
        &SM::Matrix::currentIndex,
        "Returns matrix of potential indices, given the current strain.")

    .def("currentYieldLeft",
        &SM::Matrix::currentYieldLeft,
        "Returns matrix of yield strains to the left, given the current strain.")

    .def("currentYieldRight",
        &SM::Matrix::currentYieldRight,
        "Returns matrix of yield strains to the left, given the current strain.")

    .def("Epsp",
        &SM::Matrix::Epsp,
        "Returns matrix of equivalent plastic strains, given the current strain.")

    .def("Energy",
        &SM::Matrix::Energy,
        "Returns matrix of energies, given the current strain.")

    .def("__repr__", [](const SM::Matrix&) {
        return "<GMatElastoPlasticQPot.Cartesian2d.Matrix>";
    });

}
