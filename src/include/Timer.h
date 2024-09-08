#pragma once
#include <string>
#include <chrono>
#include <iostream>


class Timer {
  
  std::string str;
  std::chrono::time_point<std::chrono::steady_clock> start, end;

public:
  Timer();
  Timer(std::string);

  void open(std::string);
  void tick();
  void tock();

};
