import Foundation

public final class KernelyraClient {
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private var requestID: UInt64 = 0
    private let lock = NSLock()

    public init(workspace: String, executable: String = "/usr/bin/env") throws {
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = executable == "/usr/bin/env"
            ? ["kernelyra", "--workspace", workspace, "rpc"]
            : ["--workspace", workspace, "rpc"]
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        try process.run()
        let ready = try readObject()
        guard ready["protocol"] as? String == "kernelyra-jsonl/1" else {
            throw NSError(domain: "Kernelyra", code: 1, userInfo: [NSLocalizedDescriptionKey: "Incompatible protocol"])
        }
    }

    public func call(_ method: String, params: [String: Any] = [:]) throws -> Any {
        lock.lock(); defer { lock.unlock() }
        requestID += 1
        let request: [String: Any] = ["id": requestID, "method": method, "params": params]
        var data = try JSONSerialization.data(withJSONObject: request)
        data.append(0x0A)
        try input.fileHandleForWriting.write(contentsOf: data)
        let response = try readObject()
        guard response["ok"] as? Bool == true else {
            throw NSError(domain: "Kernelyra", code: 2, userInfo: [NSLocalizedDescriptionKey: response["error"] as? String ?? "Kernelyra error"])
        }
        return response["result"] as Any
    }

    public func plan(_ dataset: String, options: [String: Any] = [:]) throws -> Any {
        try call("plan", params: options.merging(["dataset": dataset]) { _, new in new })
    }
    public func train(_ dataset: String, options: [String: Any] = [:]) throws -> Any {
        try call("train", params: options.merging(["dataset": dataset]) { _, new in new })
    }
    public func finetune(_ model: String, dataset: String, options: [String: Any] = [:]) throws -> Any {
        try call("finetune", params: options.merging(["model": model, "dataset": dataset]) { _, new in new })
    }

    private func readObject() throws -> [String: Any] {
        var data = Data()
        while true {
            let byte = output.fileHandleForReading.readData(ofLength: 1)
            if byte.isEmpty { throw NSError(domain: "Kernelyra", code: 3, userInfo: [NSLocalizedDescriptionKey: "Protocol closed"])}
            if byte[0] == 0x0A { break }
            data.append(byte)
        }
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    public func close() {
        try? input.fileHandleForWriting.close()
        process.waitUntilExit()
    }

    deinit { if process.isRunning { process.terminate() } }
}
