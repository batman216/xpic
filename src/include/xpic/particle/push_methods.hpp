#pragma once

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#define GET(t,i) (thrust::get<i>(t))

namespace xpic {
namespace particle {

template <typename val_type>
struct Push {

  using tuple_reference = thrust::detail::
                            tuple_of_iterator_references
                               <val_type&,val_type&,val_type&,
                                val_type&,val_type&,val_type&>;
  val_type dt;
  __host__ __device__
  Push(val_type dt) : dt(dt) {}

  __host__ __device__
  void operator()(tuple_reference t) {
    GET(t,0) = GET(t,0) + dt*(GET(t,3)+ 0*dt);
    GET(t,1) = GET(t,1) + dt*(GET(t,4)+ 0*dt);
    GET(t,2) = GET(t,2) + dt*(GET(t,5)+ 0*dt);
    GET(t,3) = GET(t,3) + 0*dt;
    GET(t,4) = GET(t,4) + 0*dt;
    GET(t,5) = GET(t,5) + 0*dt;
  }

};


template <typename Particles>
struct SymplecticEuler {

  using val_type = Particles::value_t;
  val_type dt;

  SymplecticEuler(Particles *p,val_type dt) : dt(dt) {}
    
  template <typename Cells>
  void push(Particles* p, Cells *c) {
    
    auto n = p->x[0].size();
    auto zip_itor = thrust::make_zip_iterator(p->x[0].begin(),
                                              p->x[1].begin(),
                                              p->x[2].begin(),
                                              p->v[0].begin(),
                                              p->v[1].begin(),
                                              p->v[2].begin());
    thrust::for_each(zip_itor, zip_itor+n, Push<val_type>(dt));

  }

};

} // namespace xpic
} // namespace particle
