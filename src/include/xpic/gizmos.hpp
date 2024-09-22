
template <typename T>
auto sumArray(T &arr) {
  using val_type = T::value_type;
  return std::accumulate(arr.begin(),arr.end(),0);
}

template <typename T>
void printArray(T &arr) {
  using val_type = T::value_type;
  thrust::copy(arr.begin(),arr.end(),
               std::ostream_iterator<val_type>(std::cout," "));
}
