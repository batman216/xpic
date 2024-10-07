#pragma once 
#include <map>
#include <thrust/iterator/zip_iterator.h>

namespace xpic {
  namespace particle {

    template<typename val_type,std::size_t xdim,std::size_t vdim,
             template<typename> typename Container>
    struct Particles {

      using value_t = val_type;
      const double mass, charge;
      
      std::size_t np; // # particles
      
      static const std::size_t x_dimension = xdim;
      static const std::size_t v_dimension = vdim;
      
      std::unordered_map<std::size_t,Container<val_type>*> x,v;
      Container<val_type> gamma;



      Particles(double mass, double charge) : mass(mass), charge(charge) {

        for (std::size_t i=0; i<xdim; i++)
          x[i] = new Container<val_type>();
        for (std::size_t i=0; i<vdim; i++)
          v[i] = new Container<val_type>();

      }
      
      auto particle_zip_iterator() {
        if constexpr (xdim==3) {
            return thrust::make_zip_iterator(x[0]->begin(),
                                             x[1]->begin(),
                                             x[2]->begin(),
                                             v[0]->begin(),
                                             v[1]->begin(),
                                             v[2]->begin());
        }
        if constexpr (xdim==2) {
          if constexpr (vdim==2) 
            return thrust::make_zip_iterator(x[0]->begin(),
                                             x[1]->begin(),
                                             v[0]->begin(),
                                             v[1]->begin(),
                                             v[2]->begin());
          if constexpr (vdim==3) 
            return thrust::make_zip_iterator(x[0]->begin(),
                                             x[1]->begin(),
                                             v[0]->begin(),
                                             v[1]->begin(),
                                             v[2]->begin());
        }
        if constexpr (xdim==1) {
          if constexpr (vdim==1) 
            return thrust::make_zip_iterator(x[0]->begin(),v[0]->begin());
          if constexpr (vdim==2) 
            return thrust::make_zip_iterator(x[0]->begin(),v[0]->begin(),
                                             v[1]->begin());
          if constexpr (vdim==3) 
            return thrust::make_zip_iterator(x[0]->begin(),v[0]->begin(),
                                             v[1]->begin(),v[2]->begin());
        }
      }

    };

  } // namespace particle
} // namespace xpic
