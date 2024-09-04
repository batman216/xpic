#pragma once

#include <type_traits>
#include <limits>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <initializer_list>
#include <iostream>
#include <chrono>

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

struct DoNotFlip {
  template <typename idx_type>
  __host__ __device__
  idx_type static cal(idx_type stride, std::size_t chunk, const idx_type& idx) {
    std::size_t pos = idx/chunk;
    return pos * stride + idx-(pos*chunk);
  }
};


struct Flip {
  template <typename idx_type>
  __host__ __device__
  idx_type static cal(idx_type stride, std::size_t chunk, const idx_type& idx) {
    std::size_t pos = idx/chunk;
    return pos * stride + chunk -idx+(pos*chunk)-1;
  }
};

template <typename Iterator, 
          typename Policy = DoNotFlip>
class strided_chunk_range
{
public:

  typedef typename thrust::iterator_difference<
                        Iterator>::type difference_type;
  struct stride_functor : 
  public thrust::unary_function<difference_type,difference_type> {
    difference_type stride;
    std::size_t chunk;

    stride_functor(difference_type stride, int chunk)
    : stride(stride), chunk(chunk) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const {

      // 0,1,..., chunk-1, stride+0, stride+1,..., stride+chunk-1, 
      // 2*stride+0, 2*stride+1,..., 2*stride+chunk-1, 

      return Policy::cal(stride,chunk,i);
    }
  };

  typedef typename thrust::counting_iterator
    <difference_type> CountingIterator;
  typedef typename thrust::transform_iterator
    <stride_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator
    <Iterator,TransformIterator> PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_chunk_range(Iterator first, Iterator last, 
                      difference_type stride, int chunk)
  : first(first), last(last), stride(stride), chunk(chunk) { 
    assert(chunk<=stride); 
  }

  iterator begin(void) const {
    return PermutationIterator(first,
              TransformIterator(CountingIterator(0),
                                stride_functor(stride, chunk)));
  }
  
  iterator end(void) const
  {
    difference_type lmf = last-first;
    difference_type nfs = lmf/stride;
    difference_type rem = lmf-(nfs*stride);
    return begin() + (nfs*chunk) + ((rem<chunk)?rem:chunk);
  }

  protected:
    Iterator first;
    Iterator last;
    difference_type stride;
    int chunk;
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

namespace xpic {


  template <typename ...Args>
  struct calVal {
    static constexpr std::size_t xdim = sizeof...(Args)/2;
    using val_type = typename std::tuple_element<0, std::tuple<Args...>>::type;
    using idx_type = typename std::tuple_element<xdim, std::tuple<Args...>>::type;
    using tuple_reference = thrust::detail::
                            tuple_of_iterator_references
                            <thrust::detail::
                             tuple_of_iterator_references<
                               val_type&,val_type&,val_type&>,
                             thrust::detail::
                             tuple_of_iterator_references<
                               idx_type&,idx_type&,idx_type&,idx_type&,
                               idx_type&,idx_type&,idx_type&,idx_type&>,
                             thrust::detail::
                             tuple_of_iterator_references<
                               val_type&,val_type&,val_type&,val_type&,
                               val_type&,val_type&,val_type&,val_type&>>;
    std::tuple<Args...> hn;

    __host__ __device__
    calVal(Args...args) : hn{ args... } {}

    __host__ __device__
    void operator()(tuple_reference t)  {
      std::array<val_type,xdim> x;
        x[0] = fmod(thrust::get<0>(thrust::get<0>(t)),
                    std::get<0>(hn))/std::get<0>(hn);
        x[1] = fmod(thrust::get<1>(thrust::get<0>(t)),
                    std::get<1>(hn))/std::get<1>(hn);
        x[2] = fmod(thrust::get<2>(thrust::get<0>(t)),
                    std::get<2>(hn))/std::get<2>(hn);

        idx_type i1 = floor(std::abs(thrust::get<0>(thrust::get<0>(t)))
                            /std::get<0>(hn)),
                 i2 = floor(std::abs(thrust::get<1>(thrust::get<0>(t)))
                            /std::get<1>(hn)),
                 i3 = floor(std::abs(thrust::get<2>(thrust::get<0>(t)))
                            /std::get<2>(hn));

        idx_type n1 = std::get<xdim>(hn), n2 = std::get<xdim+1>(hn),
                 n3 = std::get<xdim+2>(hn);

        thrust::get<0>(thrust::get<1>(t)) = min(i1+i2*n1+i3*n1*n2,n1*n2*n3-1); 
        thrust::get<0>(thrust::get<2>(t)) = x[0]*x[1]*x[2]; 

        thrust::get<1>(thrust::get<1>(t)) = min((i1+1)+i2*n1+i3*n1*n2,n1*n2*n3-1); 
        thrust::get<1>(thrust::get<2>(t)) = (1.0-x[0])*x[1]*x[2]; 

        thrust::get<2>(thrust::get<1>(t)) = min((i1+1)+(i2+1)*n1+i3*n1*n2,n1*n2*n3-1); 
        thrust::get<2>(thrust::get<2>(t)) = (1.0-x[0])*(1.0-x[1])*x[2]; 

        thrust::get<3>(thrust::get<1>(t)) = min(i1+(i2+1)*n1+i3*n1*n2,n1*n2*n3-1); 
        thrust::get<3>(thrust::get<2>(t)) = x[0]*(1.0-x[1])*x[2]; 

        thrust::get<4>(thrust::get<1>(t)) = min(i1+i2*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
        thrust::get<4>(thrust::get<2>(t)) = x[0]*x[1]*(1.0-x[2]); 

        thrust::get<5>(thrust::get<1>(t)) = min((i1+1)+i2*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
        thrust::get<5>(thrust::get<2>(t)) = (1.0-x[0])*x[1]*(1.0-x[2]); 

        thrust::get<6>(thrust::get<1>(t)) = min((i1+1)+(i2+1)*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
        thrust::get<6>(thrust::get<2>(t)) = (1.0-x[0])*(1.0-x[1])*(1.0-x[2]); 

        thrust::get<7>(thrust::get<1>(t)) = min(i1+(i2+1)*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
        thrust::get<7>(thrust::get<2>(t)) = x[0]*(1.0-x[1])*(1.0-x[2]); 

    }

  };

  template <typename Particle, typename Cell>
  struct Interpolate {

    static const std::size_t xdim = Particle::x_dimension;

    using val_type = Particle::value_t;
    using idx_type = std::size_t;

    cudaDataType_t cudaDataType = cudaDataTypeTraits<val_type>::type();
    cusparseIndexType_t cusparseIndexType = cusparseIndexTypeTraits<idx_type>::type(); 

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    const val_type       alpha = 1., beta = 0.;

    thrust::device_vector<idx_type> A_rows, A_cols;
    thrust::device_vector<val_type> A_vals;
    thrust::device_vector<val_type> X,Y;

    idx_type n_g, n_p, n_rows, n_cols, n_nz;    

    Particle *particles;
    Cell *cells;

    Interpolate(Particle* particles, Cell* cells) 
    : particles(particles), cells(cells) {

      n_g = cells->n_cell_tot;
      n_p = particles->x[0].size();

      n_rows = n_g;
      n_cols = n_p;
      n_nz   = n_p*std::pow(2,xdim);    

      A_rows.resize(n_nz);
      A_cols.resize(n_nz);
      A_vals.resize(n_nz);
      X.resize(n_cols); Y.resize(n_rows);

      thrust::sequence(X.begin(),X.end(),1.0,0.0);

      cusparseCreate(&handle);
      cusparseCreateCoo(&matA, n_rows, n_cols, n_nz,
                        thrust::raw_pointer_cast(A_rows.data()),
                        thrust::raw_pointer_cast(A_cols.data()),
                        thrust::raw_pointer_cast(A_vals.data()),
                        cusparseIndexType,
                        CUSPARSE_INDEX_BASE_ZERO,
                        cudaDataType);
      cusparseCreateDnVec(&vecX, n_cols,    
                          thrust::raw_pointer_cast(X.data()),cudaDataType);
      cusparseCreateDnVec(&vecY, n_rows, 
                          thrust::raw_pointer_cast(Y.data()),cudaDataType);

      cudaMalloc(&dBuffer,bufferSize); 
    }

    void go() {
      // index of particles
      idx_type dot = pow(2,xdim);
      thrust::transform(thrust::make_counting_iterator((idx_type)0),
                        thrust::make_counting_iterator(n_nz),
                        A_cols.begin(),[dot]__host__ __device__(idx_type idx) 
                        { return idx/dot; });
     
      if constexpr (xdim==3) {

        val_type d1,d2,d3; 
        idx_type n1,n2,n3;
        n1 = cells->n_cell[0];
        n2 = cells->n_cell[1];
        n3 = cells->n_cell[2];
        d1 = (cells->upper_bound[0]-cells->lower_bound[0])/n1; 
        d2 = (cells->upper_bound[1]-cells->lower_bound[1])/n2; 
        d3 = (cells->upper_bound[2]-cells->lower_bound[2])/n3; 

        auto z_itor_p = thrust::make_zip_iterator(particles->x[0].begin(),
                                                  particles->x[1].begin(),
                                                  particles->x[2].begin());

        auto z_itor_row = thrust::
                            make_zip_iterator(A_rows.begin()+0*n_p,A_rows.begin()+1*n_p,
                                              A_rows.begin()+2*n_p,A_rows.begin()+3*n_p,
                                              A_rows.begin()+4*n_p,A_rows.begin()+5*n_p,
                                              A_rows.begin()+6*n_p,A_rows.begin()+7*n_p);
        auto z_itor_val = thrust::
                            make_zip_iterator(A_vals.begin()+0*n_p,A_vals.begin()+1*n_p,
                                              A_vals.begin()+2*n_p,A_vals.begin()+3*n_p,
                                              A_vals.begin()+4*n_p,A_vals.begin()+5*n_p,
                                              A_vals.begin()+6*n_p,A_vals.begin()+7*n_p);

        auto zz_itor = thrust::make_zip_iterator(z_itor_p,z_itor_row,z_itor_val);

        
        thrust::for_each(zz_itor,zz_itor+n_p,calVal(d1,d2,d3,n1,n2,n3));
      }

      // cuSPARSE requires that A_rows must be sorted.
      auto A_cv_begin = thrust::make_zip_iterator(A_cols.begin(),A_vals.begin());
      thrust::sort_by_key(A_rows.begin(),A_rows.end(),A_cv_begin);
        
      // 最后一个网格不一定有粒子，因此最大行数不一定是网格总数
      idx_type n_row_max = 1+thrust::reduce(A_rows.begin(),A_rows.end(),
                                       thrust::device_vector<idx_type>::value_type(),
                                       thrust::maximum<idx_type>());
      cusparseSpMV_bufferSize(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,matA,vecX,&beta,vecY,cudaDataType,
                              CUSPARSE_SPMV_ALG_DEFAULT,&bufferSize);
      cudaMalloc(&dBuffer,bufferSize); 
      
      cusparseSpMV(handle,
                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                   &alpha,matA,vecX,&beta,vecY,cudaDataType,
                   CUSPARSE_SPMV_ALG_DEFAULT,dBuffer);

    }

    ~Interpolate() {
      cusparseDestroySpMat(matA); cusparseDestroyDnVec(vecX); 
      cusparseDestroyDnVec(vecY); cusparseDestroy(handle); 
    }   

  };

  template <typename Particle, typename Field>
  void cell2particle(Field* field,
                     Particle* particles) { 

       
  }


  template <typename Particle, typename Field>
  void particle2cell(Particle* particles, 
                     Field* field) {

       
  }

} // namespace xpic
