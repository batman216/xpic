#pragma once

#include <type_traits>
#include <limits>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <initializer_list>
#include <iostream>
#include <chrono>

#include "../Timer.h"
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/shuffle.h>
#include <thrust/fill.h>
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

#define SGET(t,i) (std::get<i>(t))
#define TGET(t,i) (thrust::get<i>(t))
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

      x[0] = fmod(TGET(TGET(t,0),0),SGET(hn,0))/SGET(hn,0);
      x[1] = fmod(TGET(TGET(t,0),1),SGET(hn,1))/SGET(hn,1);
      x[2] = fmod(TGET(TGET(t,0),2),SGET(hn,2))/SGET(hn,2);

      idx_type i1 = floor(std::abs(TGET(TGET(t,0),0))/SGET(hn,0)),
               i2 = floor(std::abs(TGET(TGET(t,0),1))/SGET(hn,1)),
               i3 = floor(std::abs(TGET(TGET(t,0),2))/SGET(hn,2));

      idx_type n1 = SGET(hn,xdim), n2 = SGET(hn,xdim+1), n3 = SGET(hn,xdim+2);

      TGET(TGET(t,1),0) = min(i1+i2*n1+i3*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),0) = x[0]*x[1]*x[2]; 

      TGET(TGET(t,1),1) = min((i1+1)+i2*n1+i3*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),1) = (1.0-x[0])*x[1]*x[2]; 

      TGET(TGET(t,1),2) = min((i1+1)+(i2+1)*n1+i3*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),2) = (1.0-x[0])*(1.0-x[1])*x[2]; 

      TGET(TGET(t,1),3) = min(i1+(i2+1)*n1+i3*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),3) = x[0]*(1.0-x[1])*x[2]; 

      TGET(TGET(t,1),4) = min(i1+i2*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),4) = x[0]*x[1]*(1.0-x[2]); 

      TGET(TGET(t,1),5) = min((i1+1)+i2*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),5) = (1.0-x[0])*x[1]*(1.0-x[2]); 

      TGET(TGET(t,1),6) = min((i1+1)+(i2+1)*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),6) = (1.0-x[0])*(1.0-x[1])*(1.0-x[2]); 

      TGET(TGET(t,1),7) = min(i1+(i2+1)*n1+(i3+1)*n1*n2,n1*n2*n3-1); 
      TGET(TGET(t,2),7) = x[0]*(1.0-x[1])*(1.0-x[2]); 

    }

  };

  template <typename Particle, typename Cell>
  struct Interpolate {

    Timer t1, t2;
    static const std::size_t xdim = Particle::x_dimension;

    using val_type = Particle::value_t;
    using idx_type = std::size_t;
    using ValItor  = thrust::device_vector<val_type>::iterator;
    using IdxItor  = thrust::device_vector<idx_type>::iterator;

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

    using ParticleZipper = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,ValItor>>;
    using ColZipper      = thrust::zip_iterator<thrust::tuple<IdxItor,IdxItor,IdxItor,IdxItor,
                                                              IdxItor,IdxItor,IdxItor,IdxItor>>;
    using ValZipper      = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,ValItor,ValItor,
                                                              ValItor,ValItor,ValItor,ValItor>>;
    using IntpZipper     = thrust::zip_iterator<thrust::tuple<ParticleZipper,ColZipper,ValZipper>>;

    ParticleZipper z_itor_p;
    ColZipper      z_itor_col;
    ValZipper      z_itor_val;
    IntpZipper     zz_itor;

    idx_type n_g, n_p, n_rows, n_cols, n_nz;    

    Particle *particles;
    Cell *cells;

    Interpolate(Particle* particles, Cell* cells) 
    : particles(particles), cells(cells) {

      t1.open("prepare the matrix");
      t2.open("cuSPARSE");

      n_g = cells->n_cell_tot;
      n_p = particles->x[0].size();

      n_rows = n_g;
      n_cols = n_p;
      n_nz   = n_p*std::pow(2,xdim);    

      A_rows.resize(n_nz);
      A_cols.resize(n_nz);
      A_vals.resize(n_nz);
      X.resize(n_cols); Y.resize(n_rows);

      std::puts("Interpolater good");
      using thrust::make_zip_iterator;
      z_itor_p   = make_zip_iterator(particles->x[0].begin(),
                                     particles->x[1].begin(),
                                     particles->x[2].begin());
      z_itor_col = make_zip_iterator(A_cols.begin()+0*n_p,A_cols.begin()+1*n_p,
                                     A_cols.begin()+2*n_p,A_cols.begin()+3*n_p,
                                     A_cols.begin()+4*n_p,A_cols.begin()+5*n_p,
                                     A_cols.begin()+6*n_p,A_cols.begin()+7*n_p);
      z_itor_val = make_zip_iterator(A_vals.begin()+0*n_p,A_vals.begin()+1*n_p,
                                     A_vals.begin()+2*n_p,A_vals.begin()+3*n_p,
                                     A_vals.begin()+4*n_p,A_vals.begin()+5*n_p,
                                     A_vals.begin()+6*n_p,A_vals.begin()+7*n_p);
      zz_itor    = make_zip_iterator(z_itor_p,z_itor_col,z_itor_val);
      
      cusparseCreate(&handle);
      cusparseCreateCoo(&matA, n_cols, n_rows, n_nz,
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

      thrust::fill(X.begin(),X.end(),1.0);
    }

    void go() {

      t1.tick();
      idx_type dot = pow(2,xdim);
      // index of particles
      // cuSPARSE requires that A_rows must be sorted.
      thrust::transform(thrust::make_counting_iterator((idx_type)0),
                        thrust::make_counting_iterator(n_nz),
                        A_rows.begin(),[dot]__host__ __device__(idx_type idx) 
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
        
        thrust::for_each(zz_itor,zz_itor+n_p,calVal(d1,d2,d3,n1,n2,n3));
      }

      cudaDeviceSynchronize();
      t1.tock();
      // 最后一个网格不一定有粒子，因此最大行数不一定是网格总数

      t2.tick();
      cusparseSpMV_bufferSize(handle,
                              CUSPARSE_OPERATION_TRANSPOSE,
                              &alpha,matA,vecX,&beta,vecY,cudaDataType,
                              CUSPARSE_SPMV_ALG_DEFAULT,&bufferSize);
      cudaMalloc(&dBuffer,bufferSize); 
      cusparseSpMV(handle,CUSPARSE_OPERATION_TRANSPOSE,
                   &alpha,matA,vecX,&beta,vecY,cudaDataType,
                   CUSPARSE_SPMV_ALG_DEFAULT,dBuffer);
      cudaDeviceSynchronize();
      t2.tock();
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
