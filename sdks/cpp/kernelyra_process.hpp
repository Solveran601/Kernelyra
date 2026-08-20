#pragma once
#include "kernelyra_client.hpp"
#include <memory>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace kernelyra {
namespace detail {
class RpcProcess {
  std::mutex mutex_;
#ifdef _WIN32
  HANDLE process_ = nullptr, input_ = nullptr, output_ = nullptr;
  HANDLE child_input_ = nullptr, child_output_ = nullptr;
  static std::string win_error(const char* text) { return std::string(text) + " (Win32 " + std::to_string(GetLastError()) + ')'; }
  static std::string win_arg(const std::string& value) {
    std::string result = "\""; std::size_t slashes = 0;
    for (const char character : value) {
      if (character == '\\') { ++slashes; continue; }
      if (character == '"') result.append(slashes * 2 + 1, '\\');
      else result.append(slashes, '\\');
      slashes = 0; result += character;
    }
    result.append(slashes * 2, '\\'); return result + '"';
  }
#else
  pid_t pid_ = -1; int input_ = -1, output_ = -1;
#endif
  void shutdown() noexcept {
#ifdef _WIN32
    if (child_input_) { CloseHandle(child_input_); child_input_ = nullptr; }
    if (child_output_) { CloseHandle(child_output_); child_output_ = nullptr; }
    if (input_) { CloseHandle(input_); input_ = nullptr; }
    if (output_) { CloseHandle(output_); output_ = nullptr; }
    if (process_) {
      if (WaitForSingleObject(process_, 5000) == WAIT_TIMEOUT) TerminateProcess(process_, 1);
      CloseHandle(process_); process_ = nullptr;
    }
#else
    if (input_ >= 0) { close(input_); input_ = -1; }
    if (output_ >= 0) { close(output_); output_ = -1; }
    if (pid_ > 0) {
      int status = 0;
      if (waitpid(pid_, &status, WNOHANG) == 0) { kill(pid_, SIGTERM); waitpid(pid_, &status, 0); }
      pid_ = -1;
    }
#endif
  }
  std::string line() {
    std::string value; char byte;
    while (true) {
#ifdef _WIN32
      DWORD count = 0; if (!ReadFile(output_, &byte, 1, &count, nullptr) || count == 0) throw std::runtime_error(win_error("Kernelyra closed stdout"));
#else
      const auto count = ::read(output_, &byte, 1); if (count <= 0) throw std::runtime_error("Kernelyra closed stdout");
#endif
      if (byte == '\n') return value;
      value += byte;
      if (value.size() > 1024 * 1024) throw std::runtime_error("Kernelyra response exceeds 1 MiB");
    }
  }
  void write_all(const std::string& value) {
    std::size_t offset = 0;
    while (offset < value.size()) {
#ifdef _WIN32
      DWORD count = 0; if (!WriteFile(input_, value.data() + offset, static_cast<DWORD>(value.size() - offset), &count, nullptr)) throw std::runtime_error(win_error("Cannot write Kernelyra request"));
#else
      const auto count = ::write(input_, value.data() + offset, value.size() - offset); if (count <= 0) throw std::runtime_error("Cannot write Kernelyra request");
#endif
      offset += static_cast<std::size_t>(count);
    }
  }
public:
  RpcProcess(const std::string& workspace, const std::string& executable) {
    if (executable.empty()) throw std::invalid_argument("Kernelyra executable is required");
    try {
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    if (!CreatePipe(&child_input_, &input_, &security, 0) || !CreatePipe(&output_, &child_output_, &security, 0)) throw std::runtime_error(win_error("Cannot create Kernelyra pipes"));
    if (!SetHandleInformation(input_, HANDLE_FLAG_INHERIT, 0) || !SetHandleInformation(output_, HANDLE_FLAG_INHERIT, 0)) throw std::runtime_error(win_error("Cannot secure Kernelyra pipes"));
    STARTUPINFOA startup{}; startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_input_; startup.hStdOutput = child_output_; startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION info{};
    std::string command = win_arg(executable) + " --workspace " + win_arg(workspace) + " rpc";
    if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &info)) throw std::runtime_error(win_error("Cannot start Kernelyra"));
    CloseHandle(child_input_); child_input_ = nullptr;
    CloseHandle(child_output_); child_output_ = nullptr;
    CloseHandle(info.hThread); process_ = info.hProcess;
#else
    int to_child[2], from_child[2];
    if (pipe(to_child)) throw std::runtime_error("Cannot create Kernelyra input pipe");
    if (pipe(from_child)) { close(to_child[0]); close(to_child[1]); throw std::runtime_error("Cannot create Kernelyra output pipe"); }
    pid_ = fork();
    if (pid_ < 0) {
      close(to_child[0]); close(to_child[1]); close(from_child[0]); close(from_child[1]);
      throw std::runtime_error("Cannot fork Kernelyra");
    }
    if (pid_ == 0) {
      dup2(to_child[0], STDIN_FILENO); dup2(from_child[1], STDOUT_FILENO);
      close(to_child[0]); close(to_child[1]); close(from_child[0]); close(from_child[1]);
      execlp(executable.c_str(), executable.c_str(), "--workspace", workspace.c_str(), "rpc", static_cast<char*>(nullptr));
      _exit(127);
    }
    close(to_child[0]); close(from_child[1]); input_ = to_child[1]; output_ = from_child[0];
#endif
    if (line().find("\"protocol\":\"kernelyra-jsonl/1\"") == std::string::npos) throw std::runtime_error("Incompatible Kernelyra protocol");
    } catch (...) {
      shutdown(); throw;
    }
  }
  std::string request(const std::string& value) { std::lock_guard<std::mutex> lock(mutex_); write_all(value); return line(); }
  ~RpcProcess() { shutdown(); }
};
} // namespace detail

inline Client open(const std::string& workspace = ".", const std::string& executable = "kernelyra") {
  auto process = std::make_shared<detail::RpcProcess>(workspace, executable);
  return Client([process](const std::string& request) { return process->request(request); });
}
} // namespace kernelyra
