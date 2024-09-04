#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>

#include "../util.hpp"

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
    typedef cell::Cells<Field,x_dimension> Cells;

    std::vector<Particles> all_particles;
    std::vector<std::size_t> np_init;
    std::size_t n_species;

    Cells cells;

    std::ifstream in_stream;


    ParticleInCell(std::string filename = "xpic.in") {

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

      }
      cells.n_cell_tot = std::accumulate(cells.n_cell.begin(),cells.n_cell.end(),
                                         1,std::multiplies<std::size_t>());
      std::cout << "#cell: " << cells.n_cell_tot << std::endl;
      in_stream.close();

      in_stream.open(filename);
      auto p_map = read_box(in_stream,"Particles");
      assign(n_species,"n_species",p_map);
      std::cout << "#particle species: " << n_species << std::endl;
      
      std::string m,c,np;
      std::stringstream sm(p_map["mass"]), sc(p_map["charge"]),
                        sn(p_map["particle_per_cell"]);
      for (std::size_t i=0; i<n_species; ++i) {
        std::getline(sm,m,',');
        std::getline(sc,c,',');
        all_particles.push_back(Particles(std::stod(m),std::stod(c)));

        std::getline(sn,np,','); // #particles in total
        np_init.push_back(std::stoi(np)*cells.n_cell_tot);
      }
      
      in_stream.close();


      
    }

  };


} // namespace xpic
