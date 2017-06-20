#include <gtest/gtest.h>

#include "place.h"
#include "darray.h"


#include <sstream>


void TestDArray(void) {
    using namespace majel;

    set_place(CpuPlace());

    //Can I construct a DArray from an Array?
    DArray darray(Array<float, 2>(make_dim(2, 3)));

    //Can I store to and read from a DArray?
    darray[make_dim(1, 1)] = 120.0f;
    ASSERT_EQUAL(static_cast<float>(darray[make_dim(1, 1)]), 120.0f);

    //Do stores and reads work with implicit conversions?
    darray[make_dim(1, 2)] = 3.0;
    ASSERT_EQUAL(static_cast<float>(darray[make_dim(1, 2)]), 3);

    //Can I print a dynamic reference?
    {
        std::stringstream ss;
        ss << darray[make_dim(1, 2)];
        ASSERT_EQUAL(ss.str(), "3");
    }

    //Can I print a dynamic value?
    {
        std::stringstream ss;
        const DArray& cdarray(darray);
        ss << cdarray[make_dim(1, 1)];
        ASSERT_EQUAL(ss.str(), "120");
    }

    //Can I do this all on the GPU?

    set_place(GpuPlace(0));

    //Can I construct a DArray from an Array?
    darray = DArray(Array<float, 2>(make_dim(2, 3)));

    //Can I store to and read from a DArray?
    darray[make_dim(1, 1)] = 120.0f;
    ASSERT_EQUAL(static_cast<float>(darray[make_dim(1, 1)]), 120.0f);

    set(darray, make_dim(0, 1), 56.0f);
    ASSERT_EQUAL(get(darray, make_dim(0, 1)), 56.0f);

    //Do stores and reads work with implicit conversions?
    darray[make_dim(1, 2)] = 3.0;
    ASSERT_EQUAL(static_cast<float>(darray[make_dim(1, 2)]), 3);

    //Can I print a dynamic reference?
    {
        std::stringstream ss;
        ss << static_cast<float>(darray[make_dim(1, 2)]);
        ASSERT_EQUAL(ss.str(), "3");
    }

    //Can I print a dynamic value?
    {
        std::stringstream ss;
        const DArray& cdarray(darray);
        ss << cdarray[make_dim(1, 1)];
        ASSERT_EQUAL(ss.str(), "120");
    }

    //Can I print a 1D Array
    {
        Dim<1> idx(4);

        set_place(CpuPlace());

        Array<float, 1> array(idx);

        for (int row = 0; row < 4; ++row) {
            Dim<1> dim(row);
            array[dim] = row;
        }

        std::stringstream ss;
        ss << majel::DArray(array) << std::endl;
        ASSERT_EQUAL(ss.str(), "         0 \n         1 \n         2 \n         3 \n\n");
    }

    //Can I print a 2D Array from GPU
    {
        Dim<2> idx(2, 3);

        set_place(GpuPlace(0));

        Array<float, 2> array(idx);

        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                Dim<2> dim(row, col);
                array[dim] = col;
            }
        }

        std::stringstream ss;
        ss << majel::DArray(array) << std::endl;
        ASSERT_EQUAL(ss.str(), "         0          1          2 \n         0          1          2 \n\n");
    }    
}

TEST(DArray, construct) {
  TestDArray();
}
