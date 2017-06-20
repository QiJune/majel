#include "array.h"
#include <gtest/gtest.h>

#include <vector>


void TestArray(void) {
  // Construct array on CPU
  using namespace majel;

  Dim<2> idx(1,1);
  Array<float, 2> the_array(Dim<2>(2, 3), GpuPlace(0));
  //Can I assign to Arrays on the GPU?
  the_array[idx] = 3.0f;
  ASSERT_EQ(3.0f, get(the_array, idx));


  the_array = Array<float, 2>(Dim<2>(2, 3), CpuPlace());
  // Flatten
  Array<float, 1> flat_arr = flatten(the_array);

  //Can I assign to Arrays on the CPU?
  the_array[idx] = 2.0f;
  ASSERT_EQ(2.0f, get(the_array, idx));

  Dim<2> min_idx(0, 0);
  Dim<2> max_idx(1, 2);

  // Set and read elements
  the_array[min_idx] = 5;
  ASSERT_EQ(5, the_array[min_idx]);

  the_array[max_idx] = 2;
  ASSERT_EQ(2, get(the_array, max_idx));

  set(the_array, make_dim(0, 1), 3.0f);
  ASSERT_EQ(3, the_array[make_dim(0, 1)]);

  //Test make_array
  {
    std::vector<int> tvec = {1, 0, -20, 29, 2};

    Array<int, 1> tarray = make_array(tvec);

    ASSERT_EQ(majel::get<0>(tarray.size()), static_cast<int>(tvec.size()));
    ASSERT_EQ(is_cpu_place(tarray.place()), true);
    ASSERT_EQ(is_gpu_place(tarray.place()), false);

    ASSERT_EQ(tarray[make_dim(0)], 1);
    ASSERT_EQ(tarray[make_dim(1)], 0);
    ASSERT_EQ(tarray[make_dim(2)], -20);
    ASSERT_EQ(tarray[make_dim(3)], 29);
    ASSERT_EQ(tarray[make_dim(4)], 2);

    std::vector<float> gvec = {0.01f, 3.6f, 0.0f, 20.5f, 2.53f, -700.0f};
    Array<float, 1> garray = make_array(gvec, GpuPlace());
    ASSERT_EQ(majel::get<0>(garray.size()), static_cast<int>(gvec.size()));
    ASSERT_EQ(is_cpu_place(garray.place()), false);
    ASSERT_EQ(is_gpu_place(garray.place()), true);
  }
}

TEST(Array, construct) {
  TestArray();
}
