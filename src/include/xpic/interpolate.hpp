#pragma once

#include <type_traits>
#include <limits>
#include <utility>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <initializer_list>
#include <iostream>
#include <chrono>

#include "../Timer.h"
#include "detail/xpic_traits.hpp"
#include "detail/interpolate_scheme.hpp"

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

namespace xpic {

  template <typename idx_type>
  constexpr idx_type static_pow(idx_type val,std::size_t n) {
    return  (n==0) ? 1 : (val*static_pow(val,n-1));
  }

  template <typename idx_type, typename val_type, std::size_t xdim>
  struct calVal {

    using tuple_reference = typename tupleTraits<val_type,idx_type,xdim>::tuple_reference;
    using cell_tuple = typename tupleTraits<val_type,idx_type,xdim>::cell_tuple;
    using particle_tuple = typename tupleTraits<val_type,idx_type,xdim>::particle_tuple;

    std::array<val_type,xdim> x0,dx; 
    std::array<idx_type,xdim> ng;

    __host__ __device__
    calVal(std::array<val_type,xdim> x0,
           std::array<val_type,xdim> dx,
           std::array<idx_type,xdim> ng) 
    : x0(x0),dx(dx),ng(ng) {}


    template <std::size_t...idx>
    __host__ __device__
    auto calIndex(std::index_sequence<idx...>, 
                  std::array<val_type,xdim> index,
                  std::array<idx_type,xdim> ng) {
      return thrust::make_tuple(xpic::apply(index_of_each_node<idx>(),index,ng)...);
    }

    template <std::size_t...idx>
    __host__ __device__
    auto calWeight(std::index_sequence<idx...>, 
                   std::array<val_type,xdim> wei) {
      return thrust::make_tuple(xpic::apply(weight_to_each_node<idx>(),wei)...);
    }


    __host__ __device__
    void operator()(tuple_reference t)  {

      constexpr std::size_t num_nodes = static_pow(2,xdim);

      std::array<val_type,xdim> idx,wei;
      
      wei[0] = std::modf((TGET(TGET(t,0),0)-x0[0]-0*dx[0])/dx[0],idx.data());
      if constexpr (xdim>1)
        wei[1] = std::modf((TGET(TGET(t,0),1)-x0[1]-0*dx[1])/dx[1],idx.data()+1);
      if constexpr (xdim>2) 
        wei[2] = std::modf((TGET(TGET(t,0),2)-x0[2]-0*dx[2])/dx[2],idx.data()+2);
      
      TGET(t,1)=calIndex(std::make_index_sequence<num_nodes>{},idx,ng);
      TGET(t,2)=calWeight(std::make_index_sequence<num_nodes>{},wei);

    }

  };

  template <typename Particle, typename Cell>
  struct Interpolate {

    Timer t1, t2;
    static const std::size_t xdim = Particle::x_dimension;
    static const std::size_t vdim = Particle::v_dimension;

    using val_type = Particle::value_t;
    using idx_type = std::size_t;


    cudaDataType_t cudaDataType = cudaDataTypeTraits<val_type>::type();
    cusparseIndexType_t cusparseIndexType = cusparseIndexTypeTraits<idx_type>::type(); 

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecVeclocity[vdim], vecCurrent[vdim];

    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    const val_type       alpha = 1., beta = 0.;

    thrust::device_vector<idx_type> A_rows, A_cols;
    thrust::device_vector<val_type> A_vals;
    thrust::device_vector<val_type> X,Y;

    using ParticleZipper = tupleTraits<val_type,idx_type,xdim>::ParticleZipper;
    using ColZipper      = tupleTraits<val_type,idx_type,xdim>::ColZipper;
    using ValZipper      = tupleTraits<val_type,idx_type,xdim>::ValZipper;
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

      std::puts("Interpolater good");
      using thrust::make_zip_iterator;
    }

    void go() {
      
      t1.tick();
      n_g = cells->n_cell_tot_loc_noghost;
      n_p = particles->x[0]->size();
      
      n_rows = n_g;
      n_cols = n_p;
      n_nz   = n_p*std::pow(2,xdim);    

      A_rows.resize(n_nz);
      A_cols.resize(n_nz);
      A_vals.resize(n_nz);

      z_itor_p   = make_zip_iterator(particles->x[0]->begin(),
                                     particles->x[1]->begin(),
                                     particles->x[2]->begin());
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
      for (int d=0; d<vdim; d++) {
        cusparseCreateDnVec(&vecVeclocity[d], n_cols,    
                            thrust::raw_pointer_cast(particles->v[d]->data()),
                            cudaDataType);
        auto p = cells->crnt[d].data() + cells->n_para_tot;
        cusparseCreateDnVec(&vecCurrent[d], n_rows, 
                            thrust::raw_pointer_cast(p),
                            cudaDataType);
      } 

      idx_type dot = pow(2,xdim);
      // index of particles
      // cuSPARSE requires that A_rows must be sorted.
      thrust::transform(thrust::make_counting_iterator((idx_type)0),
                        thrust::make_counting_iterator(n_nz),
                        A_rows.begin(),[dot]__host__ __device__(idx_type idx) 
                        { return idx/dot; });
     
      thrust::for_each(zz_itor,zz_itor+n_p,calVal(cells->lower_bound_loc,
                                                  cells->dx,cells->n_cell_loc_noghost));

      cudaDeviceSynchronize();
      t1.tock();

      t2.tick();
      for (int d=0; d<vdim; d++) {
        cusparseSpMV_bufferSize(handle,
                                CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha,matA,vecVeclocity[d],
                                &beta,vecCurrent[d],cudaDataType,
                                CUSPARSE_SPMV_ALG_DEFAULT,&bufferSize);
        cudaMalloc(&dBuffer,bufferSize); 
        cusparseSpMV(handle,CUSPARSE_OPERATION_TRANSPOSE,
                     &alpha,matA,vecVeclocity[d],
                     &beta,vecCurrent[d],cudaDataType,
                     CUSPARSE_SPMV_ALG_DEFAULT,dBuffer);

        // 乘上电荷
        val_type c = particles->charge;
        thrust::for_each(cells->crnt[d].begin(),cells->crnt[d].end(),
                         [c]__host__ __device__(val_type& val) { val *= c; });
      }


      cudaDeviceSynchronize();
      t2.tock();
    }

    ~Interpolate() {
      cusparseDestroySpMat(matA); 
      for (int d=0; d<vdim; d++) {
        cusparseDestroyDnVec(vecVeclocity[d]); 
        cusparseDestroyDnVec(vecCurrent[d]); 
      }
      cusparseDestroy(handle); 
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
