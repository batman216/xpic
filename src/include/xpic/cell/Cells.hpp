#pragma once 

namespace xpic {
  namespace cell {

    template<typename Field,std::size_t dim> 
    struct Cells {
  
      Field Efield, Bfield, crnt, dens;
      
      std::array<double,dim> lower_bound, upper_bound;
      std::array<std::size_t,dim> n_cell;
      std::size_t n_cell_tot;
      std::vector<Field> dens_s, crnt_s;

    };
  } // namespace cell
} // namespace xpic
