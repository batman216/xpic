#pragma once


namespace xpic {
  namespace particle {


    template <typename Particles, typename Cells, typename Method>
    class Push {
      
      Method *method;
      Push(Particles& particles, Cells& cells) {
        method->initial();
      }

      void operator()(Particles& particles, Cells& cells) {
        method->push(particles,cells);
      }

    };

  } // namespace xpic
} // namespace particle
