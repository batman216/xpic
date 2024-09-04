#pragma once 

namespace xpic {
  namespace particle {

    template<typename val_type,std::size_t xdim,std::size_t vdim,
             template<typename> typename Container>
    struct Particles {

      using value_t = val_type;
      const double mass, charge;
      
      static const std::size_t x_dimension = xdim;
      static const std::size_t v_dimension = vdim;

      std::array<Container<val_type>,xdim> x;
      std::array<Container<val_type>,vdim> v;

      Particles(double mass, double charge) : mass(mass), charge(charge) {}

    };

  } // namespace xpic
} // namespace particle
