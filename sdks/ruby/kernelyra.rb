# frozen_string_literal: true

require "json"
require "open3"
require "thread"

module Kernelyra
  class Client
    def initialize(workspace, executable: "kernelyra")
      @stdin, @stdout, @stderr, @wait = Open3.popen3(executable, "--workspace", workspace, "rpc")
      @id = 0
      @lock = Mutex.new
      ready = JSON.parse(@stdout.readline)
      raise "Incompatible Kernelyra protocol" unless ready["protocol"] == "kernelyra-jsonl/1"
    end

    def call(method, params = {})
      @lock.synchronize do
        @id += 1
        @stdin.puts(JSON.generate(id: @id, method: method, params: params))
        @stdin.flush
        response = JSON.parse(@stdout.readline)
        raise(response["error"] || "Kernelyra error") unless response["ok"]
        response["result"]
      end
    end

    def plan(dataset, **options) = call("plan", options.merge(dataset: dataset))
    def train(dataset, **options) = call("train", options.merge(dataset: dataset))
    def finetune(model, dataset, **options) = call("finetune", options.merge(model: model, dataset: dataset))

    def close
      @stdin.close unless @stdin.closed?
      @stdout.close unless @stdout.closed?
      @stderr.close unless @stderr.closed?
      @wait.value
    end
  end
end
