#pragma once
#include <memory>
#include <mutex>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kernelyra {
// Transport writes exactly one JSONL request to a persistent `kernelyra rpc`
// process and returns exactly one response line. Applications can bind this to
// Boost.Process, Qt QProcess, Win32 CreateProcess or POSIX pipes.
using Transport = std::function<std::string(const std::string&)>;

class Config {
  std::vector<std::pair<std::string, std::string>> values_;
  static std::string quote(const std::string& value) {
    std::string out = "\"";
    for (const unsigned char c : value) {
      if (c == '\\' || c == '"') { out += '\\'; out += static_cast<char>(c); }
      else if (c == '\n') out += "\\n";
      else if (c == '\r') out += "\\r";
      else if (c == '\t') out += "\\t";
      else out += static_cast<char>(c);
    }
    return out + "\"";
  }
  Config& put(const std::string& name, std::string json) {
    for (auto& item : values_) if (item.first == name) { item.second = std::move(json); return *this; }
    values_.emplace_back(name, std::move(json)); return *this;
  }
public:
  static Config automatic(const std::string& target = "") {
    Config value; if (!target.empty()) value.target(target); return value;
  }
  Config& set_json(const std::string& name, const std::string& json) { return put(name, json); }
  Config& target(const std::string& value) { return put("target", quote(value)); }
  Config& task(const std::string& value) { return put("task", quote(value)); }
  Config& backend(const std::string& value) { return put("backend", quote(value)); }
  Config& architecture(const std::string& value) { return put("architecture", quote(value)); }
  Config& model_format(const std::string& value) { return put("model_format", quote(value)); }
  Config& profile(const std::string& value) { return put("profile", quote(value)); }
  Config& goal(double value) { return put("target_metric", std::to_string(value)); }
  Config& steps(unsigned long long value) { return put("max_steps", std::to_string(value)); }
  Config& batch(unsigned value, bool accept_risk = false) {
    put("batch_size", std::to_string(value)); return put("accept_batch_risk", accept_risk ? "true" : "false");
  }
  Config& resources(unsigned cpu, unsigned ram, unsigned gpu = 0) {
    put("cpu", std::to_string(cpu)); put("ram", std::to_string(ram)); return put("gpu", std::to_string(gpu));
  }
  Config& optimizer(double learning_rate, double weight_decay = 0) {
    put("learning_rate", std::to_string(learning_rate)); return put("weight_decay", std::to_string(weight_decay));
  }
  Config& model(std::vector<unsigned> layers, const std::string& precision = "auto") {
    if (!layers.empty()) {
      std::string json = "[";
      for (std::size_t index = 0; index < layers.size(); ++index) {
        if (index) json += ',';
        json += std::to_string(layers[index]);
      }
      put("hidden_layers", json + "]");
    }
    return put("precision", quote(precision));
  }
  Config& data(unsigned workers, unsigned prefetch = 1) {
    put("data_workers", std::to_string(workers)); return put("prefetch", std::to_string(prefetch));
  }
  Config& quality(unsigned interval, double min_improvement = 0.0005,
                  unsigned early_stopping_patience = 18, unsigned target_patience = 3) {
    put("evaluation_interval", std::to_string(interval));
    put("min_improvement", std::to_string(min_improvement));
    put("early_stopping_patience", std::to_string(early_stopping_patience));
    return put("target_patience", std::to_string(target_patience));
  }
  Config& guard(double margin = 0.03, unsigned patience = 3) {
    put("degradation_margin", std::to_string(margin));
    return put("degradation_patience", std::to_string(patience));
  }
  Config& seed(unsigned long long value) { return put("seed", std::to_string(value)); }
  std::string json(const std::vector<std::pair<std::string, std::string>>& extra = {}) const {
    auto all = values_;
    for (const auto& value : extra) {
      bool replaced = false;
      for (auto& item : all) if (item.first == value.first) { item.second = value.second; replaced = true; break; }
      if (!replaced) all.push_back(value);
    }
    std::string out = "{";
    for (std::size_t index = 0; index < all.size(); ++index) {
      if (index) out += ',';
      out += quote(all[index].first) + ':' + all[index].second;
    }
    return out + '}';
  }
  static std::string string_json(const std::string& value) { return quote(value); }
};

struct TrainingResult {
  std::string json;
  std::string checkpoint() const { return string_field("checkpoint"); }
  std::string status() const { return string_field("status"); }
private:
  std::string string_field(const std::string& field) const {
    const auto marker = "\"" + field + "\":\"";
    auto at = json.find(marker); if (at == std::string::npos) return {};
    at += marker.size(); std::string out; bool escaped = false;
    for (; at < json.size(); ++at) {
      const char c = json[at];
      if (escaped) { out += c == 'n' ? '\n' : c == 'r' ? '\r' : c == 't' ? '\t' : c; escaped = false; }
      else if (c == '\\') escaped = true;
      else if (c == '"') break;
      else out += c;
    }
    return out;
  }
};

class Client {
  struct State {
    explicit State(Transport value) : transport(std::move(value)) {}
    Transport transport;
    std::mutex mutex;
    unsigned long long id = 0;
  };
  std::shared_ptr<State> state_;
  static std::string escape(const std::string& value) {
    std::string out;
    for (char c : value) {
      if (c == '\\' || c == '"') { out += '\\'; out += c; }
      else if (c == '\n') out += "\\n";
      else if (c == '\r') out += "\\r";
      else if (c == '\t') out += "\\t";
      else out += c;
    }
    return out;
  }
public:
  explicit Client(Transport transport) : state_(std::make_shared<State>(std::move(transport))) {
    if (!state_->transport) throw std::invalid_argument("Kernelyra transport is required");
  }
  std::string call(const std::string& method, const std::string& params_json = "{}") {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto request = "{\"id\":" + std::to_string(++state_->id) + ",\"method\":\"" + escape(method) + "\",\"params\":" + params_json + "}\n";
    auto response = state_->transport(request);
    if (response.find("\"ok\":true") == std::string::npos) throw std::runtime_error("Kernelyra request failed: " + response);
    return response;
  }
  std::string plan(const std::string& dataset) { return call("plan", "{\"dataset\":\"" + escape(dataset) + "\"}"); }
  std::string train(const std::string& dataset) { return call("train", "{\"dataset\":\"" + escape(dataset) + "\"}"); }
  TrainingResult fit(const std::string& dataset, const std::string& target = "", const Config& config = {}) {
    std::vector<std::pair<std::string, std::string>> extra = {{"dataset", Config::string_json(dataset)}};
    if (!target.empty()) extra.emplace_back("target", Config::string_json(target));
    return {call("train", config.json(extra))};
  }
  std::string finetune(const std::string& model, const std::string& dataset) {
    return call("finetune", "{\"model\":\"" + escape(model) + "\",\"dataset\":\"" + escape(dataset) + "\"}");
  }
  TrainingResult tune(const std::string& model, const std::string& dataset, const std::string& target = "", const Config& config = {}) {
    std::vector<std::pair<std::string, std::string>> extra = {
      {"model", Config::string_json(model)}, {"dataset", Config::string_json(dataset)}
    };
    if (!target.empty()) extra.emplace_back("target", Config::string_json(target));
    return {call("finetune", config.json(extra))};
  }
};
} // namespace kernelyra
