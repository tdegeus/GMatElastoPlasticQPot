/* =================================================================================================

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

================================================================================================= */

#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>

// Enable basic assertions on matrix shape
// (doesn't cost a lot of time, but avoids segmentation faults)
#define GMATELASTOPLASTICQPOT_ENABLE_ASSERT

// include library
#include "../include/GMatElastoPlasticQPot/Cartesian2d.h"

// abbreviate name-space
namespace py = pybind11;

// -------------------------------------------------------------------------------------------------

PYBIND11_MODULE(GMatElastoPlasticQPot, m) {

m.doc() = "Elasto-plastic material models";

// create submodule
py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");

// abbreviate name-space
namespace SM = GMatElastoPlasticQPot::Cartesian2d;

// abbreviate types(s)
typedef SM::T2 T2;

// -------------------------------------------------------------------------------------------------

sm.def("Hydrostatic",
  py::overload_cast<const T2&>(&SM::Hydrostatic),
  "Hydrostatic strain",
  py::arg("Eps"));

sm.def("Deviator",
  py::overload_cast<const T2&>(&SM::Deviator),
  "Deviator",
  py::arg("A"));

sm.def("Epsd",
  py::overload_cast<const T2&>(&SM::Epsd),
  "Equivalent strain deviator",
  py::arg("Eps"));

sm.def("Sigd",
  py::overload_cast<const T2&>(&SM::Sigd),
  "Equivalent stress deviator",
  py::arg("Sig"));

// -------------------------------------------------------------------------------------------------

sm.def("Hydrostatic",
  py::overload_cast<const xt::xtensor<double,4>&>(&SM::Hydrostatic),
  "Hydrostatic strain",
  py::arg("Eps"));

sm.def("Deviator",
  py::overload_cast<const xt::xtensor<double,4>&>(&SM::Deviator),
  "Deviator",
  py::arg("A"));

sm.def("Epsd",
  py::overload_cast<const xt::xtensor<double,4>&>(&SM::Epsd),
  "Equivalent strain deviator",
  py::arg("Eps"));

sm.def("Sigd",
  py::overload_cast<const xt::xtensor<double,4>&>(&SM::Sigd),
  "Equivalent stress deviator",
  py::arg("Sig"));

// -------------------------------------------------------------------------------------------------

py::class_<SM::Elastic>(sm, "Elastic")
  // constructor
  .def(
    py::init<double,double>(),
    "Elastic material",
    py::arg("K"),
    py::arg("G")
  )
  // methods
  .def("Stress", &SM::Elastic::Stress, py::arg("Eps"))
  .def("energy", &SM::Elastic::energy, py::arg("Eps"))
  .def("epsy"  , &SM::Elastic::epsy  , py::arg("idx"))
  .def("epsp"  , py::overload_cast<const T2&>(&SM::Elastic::epsp, py::const_), py::arg("Eps" ))
  .def("epsp"  , py::overload_cast<double   >(&SM::Elastic::epsp, py::const_), py::arg("epsd"))
  .def("find"  , py::overload_cast<const T2&>(&SM::Elastic::find, py::const_), py::arg("Eps" ))
  .def("find"  , py::overload_cast<double   >(&SM::Elastic::find, py::const_), py::arg("epsd"))
  // print to screen
  .def("__repr__", [](const SM::Elastic &){
    return "<GMatElastoPlasticQPot.Cartesian2d.Elastic>"; });

// -------------------------------------------------------------------------------------------------

py::class_<SM::Cusp>(sm, "Cusp")
  // constructor
  .def(
    py::init<double,double,const xt::xtensor<double,1>&, bool>(),
    "Cusp material",
    py::arg("K"),
    py::arg("G"),
    py::arg("epsy"),
    py::arg("init_elastic")=true
  )
  // methods
  .def("Stress", &SM::Cusp::Stress, py::arg("Eps"))
  .def("energy", &SM::Cusp::energy, py::arg("Eps"))
  .def("epsy"  , &SM::Cusp::epsy  , py::arg("idx"))
  .def("epsp"  , py::overload_cast<const T2&>(&SM::Cusp::epsp, py::const_), py::arg("Eps" ))
  .def("epsp"  , py::overload_cast<double   >(&SM::Cusp::epsp, py::const_), py::arg("epsd"))
  .def("find"  , py::overload_cast<const T2&>(&SM::Cusp::find, py::const_), py::arg("Eps" ))
  .def("find"  , py::overload_cast<double   >(&SM::Cusp::find, py::const_), py::arg("epsd"))
  // print to screen
  .def("__repr__", [](const SM::Cusp &){
    return "<GMatElastoPlasticQPot.Cartesian2d.Cusp>"; });

// -------------------------------------------------------------------------------------------------

py::class_<SM::Smooth>(sm, "Smooth")
  // constructor
  .def(
    py::init<double,double,const xt::xtensor<double,1>&, bool>(),
    "Smooth material",
    py::arg("K"),
    py::arg("G"),
    py::arg("epsy"),
    py::arg("init_elastic")=true
  )
  // methods
  .def("Stress", &SM::Smooth::Stress, py::arg("Eps"))
  .def("energy", &SM::Smooth::energy, py::arg("Eps"))
  .def("epsy"  , &SM::Smooth::epsy  , py::arg("idx"))
  .def("epsp"  , py::overload_cast<const T2&>(&SM::Smooth::epsp, py::const_), py::arg("Eps" ))
  .def("epsp"  , py::overload_cast<double   >(&SM::Smooth::epsp, py::const_), py::arg("epsd"))
  .def("find"  , py::overload_cast<const T2&>(&SM::Smooth::find, py::const_), py::arg("Eps" ))
  .def("find"  , py::overload_cast<double   >(&SM::Smooth::find, py::const_), py::arg("epsd"))
  // print to screen
  .def("__repr__", [](const SM::Smooth &){
    return "<GMatElastoPlasticQPot.Cartesian2d.Smooth>"; });

// -------------------------------------------------------------------------------------------------

py::module smm = sm.def_submodule("Type", "Type enumerator");

py::enum_<SM::Type::Value>(smm, "Type")
    .value("Unset", SM::Type::Unset)
    .value("Elastic", SM::Type::Elastic)
    .value("Cusp", SM::Type::Cusp)
    .value("Smooth", SM::Type::Smooth)
    .export_values();

// -------------------------------------------------------------------------------------------------

py::class_<SM::Matrix>(sm, "Matrix")

  .def(
    py::init<size_t, size_t>(),
    "Matrix of materials",
    py::arg("nelem"),
    py::arg("nip")
  )

  .def("type",
    &SM::Matrix::type)

  .def("nelem",
    &SM::Matrix::nelem)

  .def("nip",
    &SM::Matrix::nip)

  .def("setElastic",
    py::overload_cast<
      const xt::xtensor<size_t,2>&,
      double,
      double>(&SM::Matrix::setElastic),
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
    py::arg("I"),
    py::arg("K"),
    py::arg("G"),
    py::arg("epsy"),
    py::arg("init_elastic")=true)

  .def("setSmooth",
    py::overload_cast<
      const xt::xtensor<size_t,2>&,
      double,
      double,
      const xt::xtensor<double,1>&,
      bool>(&SM::Matrix::setSmooth),
    py::arg("I"),
    py::arg("K"),
    py::arg("G"),
    py::arg("epsy"),
    py::arg("init_elastic")=true)

  .def("setElastic",
    py::overload_cast<
      const xt::xtensor<size_t,2>&,
      const xt::xtensor<size_t,2>&,
      const xt::xtensor<double,1>&,
      const xt::xtensor<double,1>&>(&SM::Matrix::setElastic),
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
    py::arg("I"),
    py::arg("idx"),
    py::arg("K"),
    py::arg("G"),
    py::arg("epsy"),
    py::arg("init_elastic")=true)

  .def("setSmooth",
    py::overload_cast<
      const xt::xtensor<size_t,2>&,
      const xt::xtensor<size_t,2>&,
      const xt::xtensor<double,1>&,
      const xt::xtensor<double,1>&,
      const xt::xtensor<double,2>&,
      bool>(&SM::Matrix::setSmooth),
    py::arg("I"),
    py::arg("idx"),
    py::arg("K"),
    py::arg("G"),
    py::arg("epsy"),
    py::arg("init_elastic")=true)

  .def("Stress",
    py::overload_cast<const xt::xtensor<double,4>&>(&SM::Matrix::Stress, py::const_),
    py::arg("Eps"))

  .def("Energy",
    py::overload_cast<
    const xt::xtensor<double,4>&>(&SM::Matrix::Energy, py::const_),
    py::arg("Eps"))

  .def("Find",
    py::overload_cast<
    const xt::xtensor<double,4>&>(&SM::Matrix::Find, py::const_),
    py::arg("Eps"))

  .def("Epsy",
    py::overload_cast<
    const xt::xtensor<size_t,2>&>(&SM::Matrix::Epsy, py::const_),
    py::arg("idx"))

  .def("Epsp",
    py::overload_cast<
    const xt::xtensor<double,4>&>(&SM::Matrix::Epsp, py::const_),
    py::arg("Eps"))

  .def("__repr__", [](const SM::Matrix &){
    return "<GMatElastoPlasticQPot.Cartesian2d.Matrix>"; });

// -------------------------------------------------------------------------------------------------

}
