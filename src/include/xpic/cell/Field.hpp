#pragma once 

namespace xpic {
  namespace cell {

    template<typename val_type, std::size_t dim,
             template<typename> typename Container>
    struct Field {

      static const std::size_t dimension = dim;

      Container<val_type> field;

      void resize(auto n) { field.resize(n); }
      auto data() { return field.data(); }
      auto begin() const { return field.begin(); }
      auto begin() { return field.begin(); }
      auto end() const { return field.end(); }
      auto end() { return field.end(); }
    };
  } // namespace cell
} // namespace xpic
