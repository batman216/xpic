#pragma once 

namespace xpic {
  namespace cell {

    template<typename Node, std::size_t dim,
             template<typename> typename Container>
    struct Field {

      static const std::size_t dimension = dim;
      typedef Node Node_t;

      int n_cells[dim];
      Container<Node> field;

    };
  } // namespace cell
} // namespace xpic
