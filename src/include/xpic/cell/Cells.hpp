#pragma once 
#include "Field.hpp"

namespace xpic {
  namespace cell {

    template<typename val_type,std::size_t xdim, std::size_t vdim> 
    struct Cells {
  
      std::array<Field<val_type,xdim,thrust::device_vector>,vdim> Efield, Bfield, crnt;
      Field<val_type,xdim,thrust::device_vector> dens;

      std::array<double,xdim> lower_bound, upper_bound;
      std::array<double,xdim> lower_bound_loc, upper_bound_loc,dx;

      std::array<std::size_t,xdim> n_cell, n_cell_loc, n_cell_loc_noghost;
      std::size_t n_cell_tot, n_cell_tot_loc,n_cell_tot_loc_noghost, n_para_ghost=2;
      std::size_t n_para_tot;  // 每次场交互的总网格数
    };
  } // namespace cell
} // namespace xpic
