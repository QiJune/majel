#include "array.h"
#include "darray.h"
#include "dim.h"

#include <type_traits>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sstream>

#include <boost/variant.hpp>

namespace majel {

namespace {

template<typename T>
struct ContainsVisitor : boost::static_visitor<bool> {
    bool operator()(T) const {
        return true;
    }

    template<typename U>
    bool operator()(U) const {
        return false;
    }
};

struct getDValueTypeString : boost::static_visitor<std::string> {
    std::string operator()(const majel::float16&) const {
        return "float16";
    }
    std::string operator()(const float&) const {
        return "float";
    }
    std::string operator()(const double&) const {
        return "double";
    }
};

struct getDValueSize : boost::static_visitor<size_t> {
    size_t operator()(const majel::float16&) const {
        return sizeof(float16);
    }
    size_t operator()(const float&) const {
        return sizeof(float);
    }
    size_t operator()(const double&) const {
        return sizeof(double);
    }
};


struct DynamicConvert
    : public boost::static_visitor<DValue> {
    DValue value;

    DynamicConvert(DValue _value) : value(_value) {}

    template<typename R>
    DValue operator()(R) const {
        return boost::apply_visitor(ConvertDValue<R>(),
                                    value);
    }
};


}

bool is_a_float16(const DValue &val) {
    return boost::apply_visitor(ContainsVisitor<float16>(), val);
}

bool is_a_float(const DValue &val) {
    return boost::apply_visitor(ContainsVisitor<float>(), val);
}

std::string get_type_string(const DValue &val) {
    return boost::apply_visitor(getDValueTypeString(), val);
}

size_t get_dvalue_size(const DValue &val) {
    return boost::apply_visitor(getDValueSize(), val);
}


majel::DValue dynamic_convert(majel::DValue value,
                              majel::DValue type) {
    return boost::apply_visitor(DynamicConvert(value), type);
}

DReference::DReference() :
    ref(Reference<float>(PlacedPointer<float>(CpuPlace(),
                                              nullptr), nullptr)) {}

DReference::DReference(float& v, std::shared_ptr<Allocation> a) :
    ref(Reference<float>(PlacedPointer<float>(CpuPlace(),
                                              &v), a)) {}

DReference::DReference(double& v, std::shared_ptr<Allocation> a):
    ref(Reference<double>(PlacedPointer<double>(CpuPlace(),
                                                &v), a)) {}

DReference::DReference(float16& v, std::shared_ptr<Allocation> a):
    ref(Reference<float16>(PlacedPointer<float16>(CpuPlace(),
                                          &v), a)) {}

DReference::DReference(const Reference<float>& o) :
    ref(o) {}

DReference::DReference(const Reference<double>& o) :
    ref(o) {}

DReference::DReference(const Reference<float16>& o) :
    ref(o) {}

///\cond HIDDEN

struct DReferenceAssign
    : public boost::static_visitor<> {

    template<typename Ref, typename T>
    typename std::enable_if<!std::is_convertible<T, typename Ref::value_type>::value, void>::type operator()(Ref& r, const T& v) const {
        std::stringstream ss;
        ss << "Tried to assign from " << typeid(T).name()
           << " to reference of type " << typeid(Ref).name();

        throw std::invalid_argument(ss.str());
    }

    void operator()(majel::Reference<majel::float16>& r, const double& v) const {
        r = majel::float16(float(v));
    }

    template<typename Ref, typename T>
    typename std::enable_if<std::is_convertible<T, typename Ref::value_type>::value, void>::type operator()(Ref& r, const T& v) const {
        r = typename Ref::value_type(v);
    }
};

///\endcond



DReference& DReference::operator=(const float& v) {
    DReferenceAssign assigner;
    DValue dval(v);
    boost::apply_visitor(assigner, ref, dval);
    return *this;
}

DReference& DReference::operator=(const double& v) {
    DReferenceAssign assigner;
    DValue dval(v);
    boost::apply_visitor(assigner, ref, dval);
    return *this;
}

DReference& DReference::operator=(const float16& v) {
    DReferenceAssign assigner;
    DValue dval(v);
    boost::apply_visitor(assigner, ref, dval);
    return *this;
}

DReference& DReference::operator=(const DValue& v) {
    DReferenceAssign assigner;
    boost::apply_visitor(assigner, ref, v);
    return *this;
}

DReference& DReference::operator=(const Reference<float>& v) {
    ref = v;
    return *this;
}

DReference& DReference::operator=(const Reference<double>& v) {
    ref = v;
    return *this;
}

DReference& DReference::operator=(const Reference<float16>& v) {
    ref = v;
    return *this;
}

///\cond HIDDEN

struct DereferenceVisitor
    : public boost::static_visitor<DReference> {

    template<typename T, int D, int E>
    DReference operator()(Array<T, D> a, const Dim<E>& d) const {
        std::stringstream ss;
        ss << "Index dimension does not match array in DReferenceVisitor, type is: " <<
               a.get_type_and_dim() << " is a gpu place: " << is_gpu_place(a.place()) <<
               " size: " << a.size() << " and dim is " << d.dimensions << "D with contents: " <<
               d;
        throw std::invalid_argument(ss.str());
    }

    template<typename T, int D>
    DReference operator()(Array<T, D> a, const Dim<D>& d) const {
        return a[d];
    }

};

struct DevalueVisitor
    : public boost::static_visitor<DValue> {

    template<typename T, int D, int E>
    DValue operator()(const Array<T, D>& a, const Dim<E>& d) const {
        std::stringstream ss;
        ss << "Index dimension does not match array in DevalueVisitor, type is: " <<
               a.get_type_and_dim() << " is a gpu place: " << is_gpu_place(a.place()) <<
               " size: " << a.size() << " and dim is " << d.dimensions << "D with contents: " <<
               d;
        throw std::invalid_argument(ss.str());
    }

    template<typename T, int D>
    DValue operator()(const Array<T, D>& a, const Dim<D>& d) const {
        return a[d];
    }
};

struct ExtentVisitor
    : public boost::static_visitor<DDim> {

    template<typename A>
    DDim operator()(const A& a) const {
        return DDim(a.size());
    }
};

struct StrideVisitor
    : public boost::static_visitor<DDim> {

    template<typename A>
    DDim operator()(const A& a) const {
        return DDim(a.stride());
    }
};


struct AllocationVisitor
    : public boost::static_visitor<std::shared_ptr<Allocation> > {
    template<typename A>
    std::shared_ptr<Allocation> operator()(const A& a) const {
        return a.data();
    }
};

///\endcond HIDDEN

DArray::DArray() : var(Array<float, 1>()) {}

const DValue DArray::operator[](const DDim& idx) const {
    return boost::apply_visitor(DevalueVisitor(),
                                var, idx);
}

DReference DArray::operator[](const DDim& idx) {
    return boost::apply_visitor(DereferenceVisitor(),
                                var, idx);
}

DValue get(const DArray& arr, const DDim& idx) {
    return arr[idx];
}

void set(DArray& arr, const DDim& idx, const DValue& value) {
    arr[idx] = value;
}

DDim extents(const DArray& arr) {
    return boost::apply_visitor(ExtentVisitor(),
                                arr);
}

DDim strides(const DArray& arr) {
    return boost::apply_visitor(StrideVisitor(),
                                arr);
}

std::shared_ptr<Allocation> allocation(const DArray& arr) {
    return boost::apply_visitor(AllocationVisitor(),
                                arr);
}

Place place(const DArray& arr) {
    return allocation(arr)->place();
}

///\cond HIDDEN

struct DSizeOf
    : public boost::static_visitor<int> {
    template<typename T>
    int operator()(T) const {
        return sizeof(T);
    }
};

struct DContiguousStrides
    : public boost::static_visitor<DDim> {
    template<int i>
    DDim operator()(Dim<i> extents) const {
        return contiguous_strides(extents);
    }
};

struct DArrayConstructor
    : public boost::static_visitor<DArray> {
    std::shared_ptr<Allocation> alloc;
    DDim stride;
    DArrayConstructor(std::shared_ptr<Allocation> alloc_,
                      DDim stride_) : alloc(alloc_), stride(stride_) {}

    template<int i, typename T>
    typename std::enable_if<(i < 5), DArray>::type
    operator()(Dim<i> size, T type) {
        Dim<i> s_stride = boost::get<Dim<i>>(stride);
        return Array<T, i>(alloc, size, s_stride);
    }

    template<int i, typename T>
    typename std::enable_if<(i >= 5), DArray>::type
    operator()(Dim<i> dims, T type) {
        throw std::invalid_argument("DArrays are limited to 4 dimensions");
    }
};

///\endcond

DArray make_darray(std::shared_ptr<Allocation> alloc,
                   DDim dims,
                   DDim strides,
                   DValue type) {
    DArrayConstructor ctor(alloc, strides);
    return boost::apply_visitor(ctor, dims, type);
}

DArray make_darray(std::shared_ptr<Allocation> alloc,
                   DDim dims,
                   DValue type) {
    DDim strides = boost::apply_visitor(DContiguousStrides(), dims);
    return make_darray(alloc, dims, strides, type);
}

DArray make_darray(DDim dims,
                   DDim strides,
                   DValue type,
                   Place place) {
    size_t size = product(dims) *
        boost::apply_visitor(DSizeOf(), type);
    auto alloc = std::make_shared<Allocation>(size, place);
    return make_darray(alloc, dims, strides, type);
}

DArray make_darray(DDim dims,
                   DValue type,
                   Place place) {
    size_t size = product(dims) *
        boost::apply_visitor(DSizeOf(), type);
    auto alloc = std::make_shared<Allocation>(size, place);
    return make_darray(alloc, dims, type);
}

DArray make_darray(DDim dims,
                   DValue type) {
    return make_darray(dims, type, get_place());
}

Buffer buffer(const DArray& arr) {
    return Buffer(allocation(arr));
}

///\cond HIDDEN

struct DTypeExtractor
    : public boost::static_visitor<DValue> {
    template<typename T, int i>
    DValue operator()(Array<T, i>) const {
        return T(0);
    }
};

///\endcond HIDDEN

DValue type(const DArray& arr) {
    return boost::apply_visitor(DTypeExtractor(), arr.var);
}

bool types_match(std::vector<DArray> arrays) {
    bool match = true;
    DValue the_type = type(arrays.back());
    arrays.pop_back();

    for(auto i : arrays) {
        match = match && (the_type.which() == type(i).which());
    }
    return match;
}

struct ConvertVisitor : public boost::static_visitor<> {
    DArray in, out;

    ConvertVisitor(DArray in_, DArray out_) : in(in_), out(out_) {}

    template<typename T>
    void operator()(T) const {
        unary_op(convert<T>(), in, out);
    }

};

DArray to_newtype(const DArray in, const DValue newtype) {
    DArray out = make_darray(extents(in), newtype, place(in));
    copy(in, out);
    return out;
}

///\cond HIDDEN
static majel::DDim get_index(size_t dims, size_t row, size_t column) {
    std::vector<int> index;

    if(dims > 1) {
        index.push_back(row);
        index.push_back(column);
    }
    else if(dims > 0) {
        index.push_back(column);
    }

    for(int i = 2; i < (int)dims; ++i) {
        index.push_back(0);
    }

    return majel::make_ddim(index);
}
///\endcond HIDDEN

//mainly for 2D arrays to only print the quadrants for quick sanity checking
std::string preview_array(const majel::DArray &a) {
    size_t rows    = 1;
    size_t columns = 1;

    size_t dims = vectorize(extents(a)).size();

    if (dims > 1) {
        columns = get(extents(a), 1);
        rows    = get(extents(a), 0);
    }
    else if (dims > 0) {
        columns = get(extents(a), 0);
    }

    size_t x = std::min(rows,    (size_t)10);
    size_t y = std::min(columns, (size_t)10);

    size_t row_start    = 0;
    size_t column_start = 0;

    size_t row_quarter    = x / 2 + x % 2;
    size_t column_quarter = y / 2 + y % 2;

    size_t row_third_quartile    = rows    - x / 2;
    size_t column_third_quartile = columns - y / 2;

    size_t row_end    = rows;
    size_t column_end = columns;


    std::stringstream stream;

    stream << extents(a) << "\n";

    stream << "[ ";

    for (size_t row = row_start; row < row_quarter; ++row) {
        if (row != 0) {
            stream << ",\n  ";
        }

        for (size_t column = column_start; column < column_quarter; ++column) {
            if (column != 0) {
                stream << ", ";
            }

            stream << a[get_index(dims, row, column)];
        }

        if (column_quarter < column_third_quartile) {
            stream << ", ...";
        }

        for (size_t column = column_third_quartile; column < column_end; ++column) {

            stream << ", " << a[get_index(dims, row, column)];
        }
    }

    if (row_quarter < row_third_quartile) {
        stream << "\n  ...";
    }

    for (size_t row = row_third_quartile; row < row_end; ++row) {
        stream << ",\n  ";

        for (size_t column = column_start; column < column_quarter; ++column) {
            if (column != 0) {
                stream << ", ";
            }

            stream << a[get_index(dims, row, column)];
        }

        if (column_quarter < column_third_quartile) {
            stream << ", ...";
        }

        for (size_t column = column_third_quartile; column < column_end; ++column) {

            stream << ", " << a[get_index(dims, row, column)];
        }
    }

    stream << "]\n";

    return stream.str();
}

///\cond HIDDEN

struct DValuePrinter
    : public boost::static_visitor<> {

    std::ostream& os;

    DValuePrinter(std::ostream& s) : os(s) {}

    template<typename T>
    void operator()(const T& v) {
        os << v;
    }
};

///\endcond

std::ostream& operator<<(std::ostream& os, const majel::DValue& val) {
    DValuePrinter printer(os);
    boost::apply_visitor(printer, val);
    return os;
}

std::ostream& operator<<(std::ostream& os, const majel::DReference& val) {
    DValuePrinter printer(os);
    boost::apply_visitor(printer, val.ref);
    return os;
}

//XXX Works well for up to 3D arrays, hard to follow beyond that, but I'm not sure
//    any printing routine works particularly well for high dimensional arrays
std::ostream& operator<<(std::ostream& os, const majel::DArray& val) {
    os << std::setprecision(8);

    const auto& ext = majel::extents(val);
    auto dims = majel::vectorize(ext);
    std::reverse(dims.begin(), dims.end());

    long linear_max = 1;

    std::vector<int> dim_cumul;

    for (auto& d : dims) {
        dim_cumul.push_back(linear_max);
        linear_max *= d;
    }
    std::reverse(dim_cumul.begin(), dim_cumul.end());

    for (int i = 1; i <= linear_max; ++i) {
        std::vector<int> coords;
        int coord = i - 1;
        int newline = 0;
        for (auto& d : dim_cumul) {
            coords.push_back(coord / d);
            coord = coord % d;
            if ( (d != 1 && i % d == 0) || dims.size() == 1) //print vectors as columns
                ++newline;
        }
        //XXX could make printing routine nicer by figuring out max power of 10 that appears
        os << std::setw(10) << majel::get(val, majel::make_ddim(coords)) << " ";
        std::fill_n(std::ostream_iterator<std::string>(os, "\n"), newline, std::string());
    }
    return os;
}

struct DArrayTypeInfo : public boost::static_visitor<std::string> {
    template<typename T, int D>
    std::string operator()(Array<T, D> arr) const {
        return arr.get_type_and_dim();
    }
};

std::string get_type_and_dim(DArray arr) {
    return boost::apply_visitor(DArrayTypeInfo(), arr);
}

///\cond HIDDEN
struct contiguous_visitor
    : public boost::static_visitor<bool> {
    template<typename Array>
    bool operator()(const Array& a) const {
        return a.is_contiguous();
    }
};

///\endcond

bool contiguous(majel::DArray in) {
    return boost::apply_visitor(contiguous_visitor(), in);
}

} //majel
