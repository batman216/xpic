#include <thrust/device_vector.h>

template <typename val_type>
struct mpiTypeTraits;

template <>
struct mpiTypeTraits<int> {
  static constexpr MPI_Datatype type() {
    return MPI_INT;
  }
};

template <>
struct mpiTypeTraits<double> {
  static constexpr MPI_Datatype type() {
    return MPI_DOUBLE;
  }
};

template <>
struct mpiTypeTraits<float> {
  static constexpr MPI_Datatype type() {
    return MPI_FLOAT;
  }
};

template <>
struct mpiTypeTraits<long double> {
  static constexpr MPI_Datatype type() {
    return MPI_LONG_DOUBLE;
  }
};

template <typename idx_type>
struct cusparseIndexTypeTraits {
  static constexpr cusparseIndexType_t type() {
    if constexpr (sizeof(idx_type)==4)
      return CUSPARSE_INDEX_32I;
    else if  (sizeof(idx_type)==8)
      return CUSPARSE_INDEX_64I;
  }
};


template <typename val_type>
struct cudaDataTypeTraits {
  static constexpr cudaDataType_t type() {
    if constexpr (sizeof(val_type)==4)
      return CUDA_R_32F;
    else if  (sizeof(val_type)==8)
      return CUDA_R_64F;
  }
};

template <typename val_type, typename idx_type, std::size_t xdim>
struct tupleTraits;

template <typename val_type, typename idx_type>
struct tupleTraits<val_type,idx_type,1> {

  using particle_tuple = thrust::detail::
                      tuple_of_iterator_references<val_type&>;
  using cell_tuple = thrust::detail::
                       tuple_of_iterator_references<
                         val_type&,val_type&>;


  using index_tuple = thrust::detail::
                       tuple_of_iterator_references<
                         idx_type&,idx_type&>;
 
  using tuple_reference = thrust::detail::
                            tuple_of_iterator_references
                            <particle_tuple,index_tuple,cell_tuple>;

  using ValItor  = thrust::device_vector<val_type>::iterator;
  using IdxItor  = thrust::device_vector<idx_type>::iterator;

  using ParticleZipper = thrust::zip_iterator<thrust::tuple<ValItor>>;
  using ColZipper      = thrust::zip_iterator<thrust::tuple<IdxItor,IdxItor>>;
  using ValZipper      = thrust::zip_iterator<thrust::tuple<ValItor,ValItor>>;

};


template <typename val_type, typename idx_type>
struct tupleTraits<val_type,idx_type,2> {

  using ValItor  = thrust::device_vector<val_type>::iterator;
  using IdxItor  = thrust::device_vector<idx_type>::iterator;

  using ParticleZipper = thrust::zip_iterator<thrust::tuple<ValItor,ValItor>>;
  using ColZipper      = thrust::zip_iterator<thrust::tuple<IdxItor,IdxItor,
                                                            IdxItor,IdxItor>>;
  using ValZipper      = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,
                                                            ValItor,ValItor>>;

  using particle_tuple = thrust::detail::
                      tuple_of_iterator_references<
                      val_type&,val_type&>;
  using cell_tuple = thrust::detail::
                       tuple_of_iterator_references<
                       val_type&,val_type&,val_type&,val_type&>;


  using index_tuple = thrust::detail::
                       tuple_of_iterator_references<
                       idx_type&,idx_type&,idx_type&,idx_type&>;
 
  using tuple_reference = thrust::detail::
                            tuple_of_iterator_references
                            <particle_tuple,index_tuple,cell_tuple>;
};

template <typename val_type, typename idx_type>
struct tupleTraits<val_type,idx_type,3> {

  using ValItor  = thrust::device_vector<val_type>::iterator;
  using IdxItor  = thrust::device_vector<idx_type>::iterator;

  using ParticleZipper = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,ValItor>>;
  using ColZipper      = thrust::zip_iterator<thrust::tuple<IdxItor,IdxItor,
                                                            IdxItor,IdxItor, 
                                                            IdxItor,IdxItor,
                                                            IdxItor,IdxItor>>;
  using ValZipper      = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,
                                                            ValItor,ValItor,
                                                            ValItor,ValItor,
                                                            ValItor,ValItor>>;

  using particle_tuple = thrust::detail::
                      tuple_of_iterator_references<
                      val_type&,val_type&,val_type&>;
  using cell_tuple = thrust::detail::
                       tuple_of_iterator_references<
                       val_type&,val_type&,val_type&,val_type&,
                       val_type&,val_type&,val_type&,val_type&>;

  using index_tuple = thrust::detail::
                       tuple_of_iterator_references<
                       idx_type&,idx_type&,idx_type&,idx_type&,
                       idx_type&,idx_type&,idx_type&,idx_type&>;
 
  using tuple_reference = thrust::detail::
                            tuple_of_iterator_references
                            <particle_tuple,index_tuple,cell_tuple>;
};


