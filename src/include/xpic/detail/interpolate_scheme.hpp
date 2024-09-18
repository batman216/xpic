#include <numeric>

#define SGET(t,i) (std::get<i>(t))
#define TGET(t,i) (thrust::get<i>(t))

namespace xpic {

template <typename Func,typename...Arg>
__host__ __device__
auto apply(Func func, Arg... args) {
  return func(args...);
}

// ----------------- 
// |       |       |
// |   3   |   2   |
// |       |       |
// |---------------|
// |       |       |
// |   0   |   1   |
// |       |       |
// -----------------



template <std::size_t i>
struct index_of_each_node;

template <std::size_t i>
struct weight_to_each_node;


template<>
struct index_of_each_node<0> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+idx[1]*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+idx[1]*ng[0]), ng[0]*ng[1]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]),ng[0]);
    else {}
  }
};

template<>
struct weight_to_each_node<0> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(1.0-wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return (1.0-wei[0])*(1.0-wei[1]);
    else if constexpr (wei.size()==1)
      return (1.0-wei[0]);
    else {}
  }

};


template<>
struct index_of_each_node<1> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+idx[1]*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+1+idx[1]*ng[0]), ng[0]*ng[1]);
    else if constexpr (idx.size()==1)
      return std::min(static_cast<idx_type>(idx[0]+1),ng[0]);
    else {}
  }
};

template<>
struct weight_to_each_node<1> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return wei[0]*(1.0-wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return wei[0]*(1.0-wei[1]);
    else if constexpr (wei.size()==1)
      return wei[0];
    else {}
  }

};


template<>
struct index_of_each_node<2> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+(idx[1]+1)*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+1+(idx[1]+1)*ng[0]), ng[0]*ng[1]);
    else {}
  }
};

template<>
struct weight_to_each_node<2> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (wei[0])*(wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return (wei[0])*(wei[1]);
    else {}
  }
};

template<>
struct index_of_each_node<3> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+(idx[1]+1)*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+(idx[1]+1)*ng[0]), ng[0]*ng[1]);
    else {}
  }
};

template<>
struct weight_to_each_node<3> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return (1.0-wei[0])*(wei[1]);
    else {}
  }
};


template<>
struct index_of_each_node<4> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+idx[1]*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<4> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(1.0-wei[1])*wei[2];
    else {}
  }

};


template<>
struct index_of_each_node<5> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+idx[1]*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<5> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return wei[0]*(1.0-wei[1])*wei[2];
    else if constexpr (wei.size()==2)
      return wei[0]*(1.0-wei[1]);
    else {}
  }

};


template<>
struct index_of_each_node<6> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+(idx[1]+1)*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<6> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (wei[0])*(wei[1])*wei[2];
    else {}
  }
};

template<>
struct index_of_each_node<7> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+(idx[1]+1)*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<7> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(wei[1])*wei[2];
    else {}
  }
};



} // namespace xpic
