#include "kernelyra_process.hpp"
#include <iostream>

int main(int argc, char** argv) {
  const std::string dataset = argc > 1 ? argv[1] : "train.csv";
  const std::string target = argc > 2 ? argv[2] : "label";
  const std::string workspace = argc > 3 ? argv[3] : "./workspace";
  const std::string executable = argc > 4 ? argv[4] : "kernelyra";
  const std::string backend = argc > 5 ? argv[5] : "torch";
  const auto steps = argc > 6 ? std::stoull(argv[6]) : 5000ULL;
  auto engine = kernelyra::open(workspace, executable);
  auto config = kernelyra::Config::automatic().backend(backend).goal(0.95).steps(steps);
  auto result = engine.fit(dataset, target, config);
  std::cout << "status=" << result.status() << " checkpoint=" << result.checkpoint() << '\n';
}
