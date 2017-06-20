#pragma once

#include <iostream>

#include <boost/variant.hpp>

#include "array.h"
#include "ddim.h"
#include "float16.h"

namespace majel {

/**
 * \brief A dynamically typed value
 *
 */

typedef boost::variant<float, double, float16> DValue;

template<typename T>
struct ConvertDValue : public boost::static_visitor<T> {
    template<typename R>
    T operator()(R val) const {
        return static_cast<T>(val);
    }

    T operator()(Reference<float16> ref) const {
        return T(static_cast<float16>(ref).to_float());
    }

    T operator()(float16 val) const {
        return static_cast<T>(val.to_float());
    }
};

bool is_a_float16(const DValue &val);

bool is_a_float(const DValue &val);

std::string get_type_string(const DValue &val);

size_t get_dvalue_size(const DValue &val);

/**
 * \brief Convert a DValue to a different type
 * \param value The value to be converted
 * \param type The type of the resulting value
 * \return A dynamically converted value
 *
 */

majel::DValue dynamic_convert(majel::DValue value,
                              majel::DValue type);

template<typename T>
T static_convert(majel::DValue value) {
    return boost::apply_visitor(ConvertDValue<T>(), value);
}



struct DReference {
    /**
     * \brief A dynamically typed reference
     *
     */
    typedef boost::variant<Reference<float>,
                           Reference<double>,
                           Reference<float16> > DReferenceVar;
    DReferenceVar ref;

    DReference();
    DReference(float&, std::shared_ptr<Allocation>);
    DReference(double&, std::shared_ptr<Allocation>);
    DReference(float16&, std::shared_ptr<Allocation>);
    DReference(const Reference<float>&);
    DReference(const Reference<double>&);
    DReference(const Reference<float16>&);

    DReference& operator=(const float&);
    DReference& operator=(const double&);
    DReference& operator=(const float16&);
    DReference& operator=(const DValue& v);
    DReference& operator=(const Reference<float>&);
    DReference& operator=(const Reference<double>&);
    DReference& operator=(const Reference<float16>&);

private:
    template<typename T>
    T convert() const {
        return boost::apply_visitor(ConvertDValue<T>(), ref);
    }

public:

    inline operator float() const {
        return convert<float>();
    }
    inline operator double() const {
        return convert<double>();
    }
    inline operator float16() const {
        return convert<float16>();
    }

};


namespace {

typedef boost::variant<
    Array<float, 1>,
    Array<float, 2>,
    Array<float, 3>,
    Array<float, 4>,

    Array<double, 1>,
    Array<double, 2>,
    Array<double, 3>,
    Array<double, 4>,

    Array<float16, 1>,
    Array<float16, 2>,
    Array<float16, 3>,
    Array<float16, 4> > DArrayVar;

}

/**
 * \brief A dynamically typed array
 *
 */
struct DArray {
    DArrayVar var;

    DArray();

    template<typename T, int D>
    DArray(Array<T, D> in) : var(in) {}

    template<typename T>
    DArray& operator=(T in) {
        var = in;
        return *this;
    }

    const DValue operator[](const DDim&) const;
    DReference operator[](const DDim&);

    template<typename Visitor>
    typename Visitor::result_type
    apply_visitor(Visitor& visitor) {
        return var.apply_visitor(visitor);
    }

    template<typename Visitor>
    typename Visitor::result_type
    apply_visitor(Visitor& visitor) const {
        return var.apply_visitor(visitor);
    }

};

DValue get(const DArray&, const DDim&);
void set(DArray& arr, const DDim& idx, const DValue& value);
DDim extents(const DArray&);
DDim strides(const DArray&);
std::shared_ptr<Allocation> allocation(const DArray&);
DArray make_darray(DDim, DValue, Place);
DArray make_darray(DDim, DDim, DValue, Place);
DArray make_darray(DDim, DValue);
Buffer buffer(const DArray&);
DValue type(const DArray&);
Place place(const DArray&);

DArray to_newtype(const DArray, const DValue);

bool contiguous(const DArray);

std::string get_type_and_dim(DArray arr);

/**
 * \brief Returns true if the DArray objects have the same type
 */

bool types_match(std::vector<DArray>);

std::string preview_array(const majel::DArray& a);

template<typename T, int N>
std::string preview_array(const majel::Array<T, N>& a) {
    return preview_array(majel::DArray(a));
}

std::ostream& operator<<(std::ostream& os, const majel::DValue& v);
std::ostream& operator<<(std::ostream& os, const majel::DReference& v);
std::ostream& operator<<(std::ostream& os, const majel::DArray& v);

}

template<typename T, int N>
std::ostream& operator<<(std::ostream& os, const majel::Array<T, N>& v) {
    os << majel::DArray(v);
    return os;
}


namespace boost {
template<typename T>
T get(const majel::DArray& in) {
    return boost::get<T>(in.var);
}
}
