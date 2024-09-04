#pragma once
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>


namespace xpic {

  template <typename val_type>
  struct randgen { // generate uniform random numbers

    val_type l,u; unsigned int seed;
    __host__ __device__
    randgen(unsigned int s, val_type l=0, val_type u=0):seed(s),l(l),u(u) {};

    __host__ __device__
    val_type operator()(const std::size_t n) const {
      // DONOT use the default engine
      thrust::random::taus88 rng(seed);
      thrust::random::uniform_real_distribution<val_type> dist(l,u);
      rng.discard(n);
      return dist(rng);
    }
  };

  template <typename val_type>
  struct randngen { // generate Gaussian random numbers

    val_type w,x; unsigned int seed;
    __host__ __device__
    randngen(unsigned int s, val_type x=0, val_type w=1):seed(s),x(x),w(w) {};

    __host__ __device__
    val_type operator()(const std::size_t n) const {
      // DONOT use the default engine
      thrust::random::taus88 rng(seed);
      thrust::random::normal_distribution<val_type> dist(x,w);
      rng.discard(n);
      return dist(rng);
    }
  };

  template <typename PIC>
  void loadParticle(PIC *pic) {
    
    using Container = PIC::container;
    using val_type  = PIC::value_t;

    std::cout << pic->n_species << std::endl;
    for (std::size_t s=0; s<pic->n_species; ++s) {

      thrust::host_vector<val_type> buf(pic->np_init[s]);
      for (std::size_t d=0; d<PIC::x_dimension; ++d) {
        
        pic->all_particles[s].x[d].resize(pic->np_init[s]);
        pic->all_particles[s].v[d].resize(pic->np_init[s]);

        unsigned int seed = (d+23)*123237;
        thrust::transform(thrust::make_counting_iterator((std::size_t)0),
                          thrust::make_counting_iterator(pic->np_init[s]),
                          pic->all_particles[s].x[d].begin(), 
                          randngen(seed,10.0,2.0));
                          //randgen(seed,pic->cells.lower_bound[d],
                           //            pic->cells.upper_bound[d]/2));

        thrust::transform(thrust::make_counting_iterator((std::size_t)0),
                          thrust::make_counting_iterator(pic->np_init[s]),
                          pic->all_particles[s].v[d].begin(), 
                          randngen(seed,0.0,1.0));

      }
    }
    std::puts("particles successfully loaded.");
  }


} // namespace xpic 
