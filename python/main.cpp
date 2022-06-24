/**
\file
\copyright Copyright. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_python_config.hpp> // todo: remove for xtensor-python >0.26.1

#define GMATELASTOPLASTICQPOT_USE_XTENSOR_PYTHON
#define GMATTENSOR_USE_XTENSOR_PYTHON
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GMatElastoPlasticQPot/version.h>
#include <GMatTensor/Cartesian2d.h>

namespace py = pybind11;

template <class S, class T>
auto construct_Elastic(T& cls)
{
    cls.def(
        py::init<const xt::pytensor<double, S::rank>&, const xt::pytensor<double, S::rank>&>(),
        "Heterogeneous system.",
        py::arg("K"),
        py::arg("G"));
}

template <class S, class T>
auto construct_Cusp(T& cls)
{
    cls.def(
        py::init<
            const xt::pytensor<double, S::rank>&,
            const xt::pytensor<double, S::rank>&,
            const xt::pytensor<double, S::rank + 1>&>(),
        "Heterogeneous system.",
        py::arg("K"),
        py::arg("G"),
        py::arg("epsy"));
}

template <class S, class T>
auto setname(T& cls, const std::string& name)
{
    cls.def("__repr__", [=](const S&) { return name; });
}

template <class S, class T>
auto add_Elastic(T& cls)
{
    cls.def_property_readonly("shape", &S::shape, "Shape of array.");
    cls.def_property_readonly("shape_tensor2", &S::shape_tensor2, "Array of rank 2 tensors.");
    cls.def_property_readonly("shape_tensor4", &S::shape_tensor4, "Array of rank 4 tensors.");
    cls.def_property_readonly("K", &S::K, "Bulk modulus.");
    cls.def_property_readonly("G", &S::G, "Shear modulus.");
    cls.def_property_readonly("Sig", &S::Sig, "Stress tensor.");
    cls.def_property_readonly("C", &S::C, "Tangent tensor.");
    cls.def_property_readonly("energy", &S::energy, "Potential energy.");

    cls.def_property(
        "Eps",
        static_cast<xt::pytensor<double, S::rank + 2>& (S::*)()>(&S::Eps),
        &S::template set_Eps<xt::pytensor<double, S::rank + 2>>,
        "Strain tensor");

    cls.def("refresh", &S::refresh, "Recompute stress from strain.");
}

template <class S, class T>
auto add_Cusp(T& cls)
{
    cls.def_property(
        "epsy",
        &S::epsy,
        &S::template set_epsy<xt::pytensor<double, S::rank + 1>>,
        "Yield strain history");

    cls.def_property_readonly("i", &S::i, "Index in epsy");
    cls.def_property_readonly("eps", &S::eps, "Equivalent strain deviator");
    cls.def_property_readonly("epsy_left", &S::epsy_left, "epsy[..., i] (copy)");
    cls.def_property_readonly("epsy_right", &S::epsy_right, "epsy[..., i + 1] (copy)");
    cls.def_property_readonly("epsp", &S::epsp, "0.5 * (epsy[..., i] + epsy[..., i + 1]) (copy)");
}

template <class S, class T>
auto Elastic(T& cls)
{
    construct_Elastic<S>(cls);
    add_Elastic<S>(cls);
    setname<S>(cls, "<GMatElastoPlasticQPot.Cartesian2d.Elastic>");
}

template <class S, class T>
auto Cusp(T& cls, const std::string& name)
{
    construct_Cusp<S>(cls);
    add_Elastic<S>(cls);
    add_Cusp<S>(cls);
    setname<S>(cls, "<GMatElastoPlasticQPot.Cartesian2d." + name + ">");
}

template <class R, class T, class M>
void Epsd(M& mod)
{
    mod.def(
        "Epsd",
        static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Epsd),
        "Equivalent strain of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void epsd(M& mod)
{
    mod.def(
        "epsd",
        static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::epsd),
        "Equivalent strain of a(n) (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void Sigd(M& mod)
{
    mod.def(
        "Sigd",
        static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Sigd),
        "Equivalent stress of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void sigd(M& mod)
{
    mod.def(
        "sigd",
        static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::sigd),
        "Equivalent stress of a(n) (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

/**
Overrides the `__name__` of a module.
Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
which affects the `__repr__` of the class type objects.
*/
class ScopedModuleNameOverride {
public:
    explicit ScopedModuleNameOverride(py::module m, std::string name) : module_(std::move(m))
    {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride()
    {
        module_.attr("__name__") = original_name_;
    }

private:
    py::module module_;
    py::object original_name_;
};

PYBIND11_MODULE(_GMatElastoPlasticQPot, m)
{
    ScopedModuleNameOverride name_override(m, "GMatElastoPlasticQPot");

    xt::import_numpy();

    m.doc() = "Elasto-plastic material model";

    m.def("version", &GMatElastoPlasticQPot::version, "Version string.");

    m.def(
        "version_dependencies",
        &GMatElastoPlasticQPot::version_dependencies,
        "List of version strings, include dependencies.");

    m.def(
        "version_compiler",
        &GMatElastoPlasticQPot::version_compiler,
        "Information on the compiler, the platform, the C++ standard, and the compilation data.");

    // ---------------------------------
    // GMatElastoPlasticQPot.Cartesian2d
    // ---------------------------------

    py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");

    namespace SM = GMatElastoPlasticQPot::Cartesian2d;

    // Tensor algebra

    Epsd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    Epsd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    Epsd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    epsd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    epsd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    epsd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    Sigd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    Sigd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    Sigd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    sigd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    sigd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    sigd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    // Elastic

    py::class_<SM::Elastic<0>, GMatTensor::Cartesian2d::Array<0>> E_array0d(sm, "Elastic0d");
    py::class_<SM::Elastic<1>, GMatTensor::Cartesian2d::Array<1>> E_array1d(sm, "Elastic1d");
    py::class_<SM::Elastic<2>, GMatTensor::Cartesian2d::Array<2>> E_array2d(sm, "Elastic2d");
    py::class_<SM::Elastic<3>, GMatTensor::Cartesian2d::Array<3>> E_array3d(sm, "Elastic3d");

    Elastic<SM::Elastic<0>>(E_array0d);
    Elastic<SM::Elastic<1>>(E_array1d);
    Elastic<SM::Elastic<2>>(E_array2d);
    Elastic<SM::Elastic<3>>(E_array3d);

    // Cusp

    py::class_<SM::Cusp<0>, GMatTensor::Cartesian2d::Array<0>> C_array0d(sm, "Cusp0d");
    py::class_<SM::Cusp<1>, GMatTensor::Cartesian2d::Array<1>> C_array1d(sm, "Cusp1d");
    py::class_<SM::Cusp<2>, GMatTensor::Cartesian2d::Array<2>> C_array2d(sm, "Cusp2d");
    py::class_<SM::Cusp<3>, GMatTensor::Cartesian2d::Array<3>> C_array3d(sm, "Cusp3d");

    Cusp<SM::Cusp<0>>(C_array0d, "Cusp");
    Cusp<SM::Cusp<1>>(C_array1d, "Cusp");
    Cusp<SM::Cusp<2>>(C_array2d, "Cusp");
    Cusp<SM::Cusp<3>>(C_array3d, "Cusp");

    // Smooth

    py::class_<SM::Smooth<0>, GMatTensor::Cartesian2d::Array<0>> S_array0d(sm, "Smooth0d");
    py::class_<SM::Smooth<1>, GMatTensor::Cartesian2d::Array<1>> S_array1d(sm, "Smooth1d");
    py::class_<SM::Smooth<2>, GMatTensor::Cartesian2d::Array<2>> S_array2d(sm, "Smooth2d");
    py::class_<SM::Smooth<3>, GMatTensor::Cartesian2d::Array<3>> S_array3d(sm, "Smooth3d");

    Cusp<SM::Smooth<0>>(S_array0d, "Smooth");
    Cusp<SM::Smooth<1>>(S_array1d, "Smooth");
    Cusp<SM::Smooth<2>>(S_array2d, "Smooth");
    Cusp<SM::Smooth<3>>(S_array3d, "Smooth");
}
