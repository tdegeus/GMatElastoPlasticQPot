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


PYBIND11_MODULE(GMatElastoPlasticQPot, m) {

m.doc() = "Elasto-plastic material model";


// ---------------------------------
// GMatElastoPlasticQPot.Cartesian2d
// ---------------------------------

py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");

namespace SM = GMatElastoPlasticQPot::Cartesian2d;


// Tensor algebra


sm.def("Hydrostatic",
    py::overload_cast<const SM::Tensor2&>(&SM::Hydrostatic),
    "Hydrostatic part of a 2nd-order tensor. Returns scalar.",
    py::arg("A"));

sm.def("Hydrostatic",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Hydrostatic),
    "Hydrostatic part of a 2nd-order tensor. Returns list of scalars.",
    py::arg("A"));

sm.def("Hydrostatic",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Hydrostatic),
    "Hydrostatic part of a 2nd-order tensor. Returns matrix of scalars.",
    py::arg("A"));


sm.def("Deviatoric",
    py::overload_cast<const SM::Tensor2&>(&SM::Deviatoric),
    "Deviatoric part of a 2nd-order tensor. Returns 2nd-order tensor.",
    py::arg("A"));

sm.def("Deviatoric",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Deviatoric),
    "Deviatoric part of a 2nd-order tensor. Returns list 2nd-order tensors.",
    py::arg("A"));

sm.def("Deviatoric",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Deviatoric),
    "Deviatoric part of a 2nd-order tensor. Returns matrix 2nd-order tensors.",
    py::arg("A"));


sm.def("Epsd",
    py::overload_cast<const SM::Tensor2&>(&SM::Epsd),
    "Equivalent strain deviator. Returns scalar.",
    py::arg("Eps"));

sm.def("Epsd",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Epsd),
    "Equivalent strain deviator. Returns list of scalars.",
    py::arg("Eps"));

sm.def("Epsd",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Epsd),
    "Equivalent strain deviator. Returns matrix of scalars.",
    py::arg("Eps"));


sm.def("Sigd",
    py::overload_cast<const SM::Tensor2&>(&SM::Sigd),
    "Equivalent stress deviator. Returns scalar.",
    py::arg("Sig"));

sm.def("Sigd",
    py::overload_cast<const xt::xtensor<double,3>&>(&SM::Sigd),
    "Equivalent stress deviator. Returns list of scalars.",
    py::arg("Sig"));

sm.def("Sigd",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Sigd),
    "Equivalent stress deviator. Returns matrix of scalars.",
    py::arg("Sig"));


// Material point: Elastic


py::class_<SM::Elastic>(sm, "Elastic")

    .def(py::init<double, double>(),
        "Elastic material point",
        py::arg("K"),
        py::arg("G"))

    .def("Stress",
        &SM::Elastic::Stress,
        "Returns stress tensor, for a given strain tensor.",
        py::arg("Eps"))

    .def("energy",
        &SM::Elastic::energy,
        "Returns the energy, for a given strain tensor.",
        py::arg("Eps"))

    .def("__repr__", [](const SM::Elastic &){
        return "<GMatElastoPlasticQPot.Cartesian2d.Elastic>"; });


// Material point: Cusp


py::class_<SM::Cusp>(sm, "Cusp")

    .def(py::init<double,double,const xt::xtensor<double,1>&, bool>(),
        "Elasto-plastic material point, with 'cusp' potentials",
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"),
        py::arg("init_elastic") = true)

    .def("Stress",
        &SM::Cusp::Stress,
        "Returns stress tensor, for a given strain tensor.",
        py::arg("Eps"))

    .def("energy",
        &SM::Cusp::energy,
        "Returns the energy, for a given strain tensor.",
        py::arg("Eps"))

    .def("epsy",
        &SM::Cusp::epsy,
        "Returns yield strain for a given potential index.",
        py::arg("idx"))

    .def("epsp",
        py::overload_cast<const SM::Tensor2&>(&SM::Cusp::epsp, py::const_),
        "Returns the equivalent plastic strain for a given strain tensor.",
        py::arg("Eps"))

    .def("epsp",
        py::overload_cast<double>(&SM::Cusp::epsp, py::const_),
        "Returns the equivalent plastic strain for a given equivalent strain deviator.",
        py::arg("epsd"))

    .def("find",
        py::overload_cast<const SM::Tensor2&>(&SM::Cusp::find, py::const_),
        "Returns the potential index, for a given strain tensor.",
        py::arg("Eps"))

    .def("find",
        py::overload_cast<double>(&SM::Cusp::find, py::const_),
        "Returns the potential index, for a given equivalent strain deviator.",
        py::arg("epsd"))

    .def("__repr__", [](const SM::Cusp &){
        return "<GMatElastoPlasticQPot.Cartesian2d.Cusp>"; });


// Material point: Smooth


py::class_<SM::Smooth>(sm, "Smooth")

  .def(py::init<double,double,const xt::xtensor<double,1>&, bool>(),
      "Elasto-plastic material point, with 'smooth' potentials",
      py::arg("K"),
      py::arg("G"),
      py::arg("epsy"),
      py::arg("init_elastic") = true)

    .def("Stress",
        &SM::Smooth::Stress,
        "Returns stress tensor, for a given strain tensor.",
        py::arg("Eps"))

    .def("energy",
        &SM::Smooth::energy,
        "Returns the energy, for a given strain tensor.",
        py::arg("Eps"))

    .def("epsy",
        &SM::Smooth::epsy,
        "Returns yield strain for a given potential index.",
        py::arg("idx"))

    .def("epsp",
        py::overload_cast<const SM::Tensor2&>(&SM::Smooth::epsp, py::const_),
        "Returns the equivalent plastic strain for a given strain tensor.",
        py::arg("Eps"))

    .def("epsp",
        py::overload_cast<double>(&SM::Smooth::epsp, py::const_),
        "Returns the equivalent plastic strain for a given equivalent strain deviator.",
        py::arg("epsd"))

    .def("find",
        py::overload_cast<const SM::Tensor2&>(&SM::Smooth::find, py::const_),
        "Returns the potential index, for a given strain tensor.",
        py::arg("Eps"))

    .def("find",
        py::overload_cast<double>(&SM::Smooth::find, py::const_),
        "Returns the potential index, for a given equivalent strain deviator.",
        py::arg("epsd"))

    .def("__repr__", [](const SM::Smooth &){
        return "<GMatElastoPlasticQPot.Cartesian2d.Smooth>"; });


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

    .def("ndim",
        &SM::Matrix::ndim,
        "Return number of (tensor) dimensions.")

    .def("nelem",
        &SM::Matrix::nelem,
        "Return number of elements (matrix rows).")

    .def("nip",
        &SM::Matrix::nip,
        "Return number of integration points (matrix columns).")

    .def("K",
        &SM::Matrix::K,
        "Return matrix with bulk moduli.")

    .def("G",
        &SM::Matrix::G,
        "Return matrix with shear moduli.")

    .def("type",
        &SM::Matrix::type,
        "Return matrix with material types.")

    .def("isPlastic",
        &SM::Matrix::isPlastic,
        "Return matrix with boolean: elastic (0) or plastic (1).")

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
        py::overload_cast<
            const xt::xtensor<size_t,2>&,
            double,
            double>(&SM::Matrix::setElastic),
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

    .def("Stress",
        &SM::Matrix::Stress,
        "Returns matrix of stress tensors, given matrix of strain tensors.",
        py::arg("Eps"))

    .def("Energy",
        &SM::Matrix::Energy,
        "Returns matrix of energies, given matrix of strain tensors.",
        py::arg("Eps"))

    .def("Epsy",
        &SM::Matrix::Epsy,
        "Returns matrix of yield strains, given matrix of potential indices.",
        py::arg("idx"))

    .def("Epsp",
        &SM::Matrix::Epsp,
        "Returns matrix of equivalent plastic strains, given matrix of strain tensors.",
        py::arg("Eps"))

    .def("Find",
        &SM::Matrix::Find,
        "Returns matrix of potential indices, given matrix of strain tensors.",
        py::arg("Eps"))

    .def("__repr__", [](const SM::Matrix &){
        return "<GMatElastoPlasticQPot.Cartesian2d.Matrix>"; });

}

