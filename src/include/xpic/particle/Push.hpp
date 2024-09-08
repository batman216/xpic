#pragma once


namespace xpic {
  namespace particle {


    template <typename Method>
    class Push {

      using val_type = Particles::value_t;
      val_type dt;

      Method *method;
      template <typename Particles, typename Cells>
      Push(Particles&& particles, Cells&& cells) {
        method->initial();
      }

      template <typename Particles, typename Cells>
      void operator()(Particles&& particles, Cells&& cells) {
        method->push(particles,cells);
      }

    };

  } // namespace xpic
} // namespace particle
