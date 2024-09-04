#include "include/Timer.h"

Timer::Timer() {}


Timer::Timer(std::string str) {
  
  this->str = str;

}

void Timer::tick() {
  start  = std::chrono::steady_clock::now();
}
void Timer::tock() {
  end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << str << " time cost: " << elapsed.count() << "s" << std::endl;

}

