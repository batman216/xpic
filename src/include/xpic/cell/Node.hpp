#pragma once 

namespace xpic {
  namespace cell {

    template<typename val_type, std::size_t dim>
    struct Node {
      
      static const std::size_t dimension = dim;

      val_type val[dim];

      Node() {}
      __host__ __device__
      Node(std::initializer_list<val_type> initial_list) {
        int i=0;
        for (val_type elem : initial_list) 
          val[i++] = elem; 
      }
    };
  } // namespace xpic
} // namespace cell
