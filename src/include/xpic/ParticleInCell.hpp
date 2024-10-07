#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <nccl.h>

#include "../util.hpp"
#include "ParallelCommunicator.hpp"

#include "particle/Particles.hpp"

#include "cell/Cells.hpp"
#include "cell/Field.hpp"
#include "cell/Node.hpp"

namespace xpic {

  template<typename val_type,
           std::size_t xdim, std::size_t vdim>
  struct ParticleInCell {

    static const std::size_t x_dimension = xdim, v_dimension = vdim;
    typedef val_type value_t;
    typedef thrust::device_vector<val_type> container;
    typedef particle::Particles<val_type,xdim,vdim,
            thrust::device_vector> Particles;

    typedef cell::Node<val_type,v_dimension> Node;
    typedef cell::Field<Node,x_dimension,thrust::device_vector> Field;
    typedef cell::Cells<val_type,x_dimension,v_dimension> Cells;

    ncclComm_t comm;
    cudaStream_t s;
    ncclUniqueId nccl_id;
    ParallelCommunicator<std::size_t,val_type> *para;

    std::vector<Particles> all_particles;
    std::vector<std::size_t> np_init;
    std::vector<std::array<val_type,vdim>> v_thermal,v_drift;

    std::size_t n_species;

    const std::size_t n_mpi, r_mpi;
    const char flag_mpi;

    Cells cells;

    std::ifstream in_stream;


    ParticleInCell(ParallelCommunicator<std::size_t,val_type> *para,
                   std::string filename = "xpic.in") 
    : para(para),s(para->s), nccl_id(para->nccl_id),comm(para->comm),
      n_mpi(para->mpi_size), r_mpi(para->mpi_rank), flag_mpi(para->flag) {

      in_stream.open(filename);

      auto c_map = read_box(in_stream,"Cells");
  
      std::string ncell,lbd,ubd;
      std::stringstream cellss(c_map["n_cell"]),
         ubdss(c_map["upper_bound"]),lbdss(c_map["lower_bound"]);
      for (std::size_t d=0; d<x_dimension; ++d) {
        std::getline(cellss,ncell,',');
        cells.n_cell[d] = std::stoi(ncell);
        std::getline(lbdss,lbd,',');
        cells.lower_bound[d] = std::stod(lbd);
        std::getline(ubdss,ubd,',');
        cells.upper_bound[d] = std::stod(ubd);
        cells.dx[d] = (cells.upper_bound[d]-cells.lower_bound[d])
                       /cells.n_cell[d];
      }

      for (std::size_t d=0; d<x_dimension; ++d) {
        cells.lower_bound_loc[d] = cells.lower_bound[d];
        cells.upper_bound_loc[d] = cells.upper_bound[d];
        cells.n_cell_loc_noghost[d] = cells.n_cell[d];
        cells.n_cell_loc[d] = cells.n_cell[d];
      }

      cells.n_cell_tot = std::accumulate(cells.n_cell.begin(),cells.n_cell.end(),
                                         1,std::multiplies<std::size_t>());

      /// 只并行第xdim-1个维度
      cells.n_cell_loc_noghost[xdim-1] /= n_mpi;
      cells.n_cell_loc[xdim-1] = cells.n_cell_loc_noghost[xdim-1]
                               + cells.n_para_ghost*2;
      cells.n_cell_tot_loc = std::accumulate(cells.n_cell_loc.begin(),cells.n_cell_loc.end(),
                                             1,std::multiplies<std::size_t>());
      cells.n_cell_tot_loc_noghost = std::accumulate(cells.n_cell_loc_noghost.begin(),
                                                     cells.n_cell_loc_noghost.end(),
                                                     1,std::multiplies<std::size_t>());

      if constexpr (xdim==3)
        cells.n_para_tot = cells.n_para_ghost*cells.n_cell[1]*cells.n_cell[0];
      else if constexpr (xdim == 2)
        cells.n_para_tot = cells.n_para_ghost*cells.n_cell[0];
      else
        cells.n_para_tot = cells.n_para_ghost;



      val_type L = cells.upper_bound[xdim-1] - cells.lower_bound[xdim-1],
               l = L / n_mpi;
      cells.lower_bound_loc[xdim-1] = l* r_mpi;
      cells.upper_bound_loc[xdim-1] = l*(r_mpi+1);
        
      for (int d=0; d<vdim; d++) {
        cells.Efield[d].resize(cells.n_cell_tot_loc);
        cells.Bfield[d].resize(cells.n_cell_tot_loc);
        cells.crnt[d].resize(cells.n_cell_tot_loc);
      }


      std::cout << "#cell: " << cells.n_cell_tot_loc << std::endl;
      in_stream.close();

      in_stream.open(filename);
      auto p_map = read_box(in_stream,"Particles");

      assign(n_species,"n_species",p_map);
      std::cout << "#particle species: " << n_species << std::endl;
      
      std::string m,c,np, vt1,vt2,vt3,vd1,vd2,vd3;
      std::stringstream sm(p_map["mass"]), 
                        sc(p_map["charge"]),
                        sn(p_map["particle_per_cell"]),
                        svd1(p_map["drift_v1"]),
                        svd2(p_map["drift_v2"]),
                        svd3(p_map["drift_v3"]),
                        svt1(p_map["thermal_v1"]),
                        svt2(p_map["thermal_v2"]),
                        svt3(p_map["thermal_v3"]);

#define S2D(x) (std::stof(x))

      for (std::size_t i=0; i<n_species; ++i) {
        std::getline(sm,m,',');
        std::getline(sc,c,',');
        all_particles.push_back(Particles(S2D(m),S2D(c)));

        std::getline(sn,np,','); // #particles in total
        np_init.push_back(std::stoi(np)*cells.n_cell_tot/n_mpi);

        std::getline(svd1,vd1,','); 
        std::getline(svd2,vd2,','); 
        std::getline(svd3,vd3,','); 
        if constexpr (vdim==1)
          v_drift.push_back({S2D(vd1)});
        if constexpr (vdim==2)
          v_drift.push_back({S2D(vd1),S2D(vd2)});
        if constexpr (vdim==3)
          v_drift.push_back({S2D(vd1),S2D(vd2),S2D(vd3)});

        std::getline(svt1,vt1,','); 
        std::getline(svt2,vt2,','); 
        std::getline(svt3,vt3,','); 
        if constexpr (vdim==1)
          v_thermal.push_back({S2D(vt1)});
        if constexpr (vdim==2)
          v_thermal.push_back({S2D(vt1),S2D(vt2)});
        if constexpr (vdim==3)
          v_thermal.push_back({S2D(vt1),S2D(vt2),S2D(vt3)});

      }
      
      in_stream.close();

    }

  };


} // namespace xpic
