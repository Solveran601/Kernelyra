package kernelyra

import java.io.Closeable
import java.util.concurrent.atomic.AtomicLong
import kotlinx.serialization.json.*

class KernelyraClient(workspace: String, executable: String = "kernelyra") : Closeable {
    private val process = ProcessBuilder(executable, "--workspace", workspace, "rpc").redirectError(ProcessBuilder.Redirect.INHERIT).start()
    private val input = process.outputStream.bufferedWriter(Charsets.UTF_8)
    private val output = process.inputStream.bufferedReader(Charsets.UTF_8)
    private val ids = AtomicLong()

    init {
        val ready = Json.parseToJsonElement(output.readLine()).jsonObject
        require(ready["protocol"]?.jsonPrimitive?.content == "kernelyra-jsonl/1") { "Incompatible Kernelyra protocol" }
    }

    @Synchronized fun call(method: String, params: JsonObject = buildJsonObject {}): JsonElement {
        val request = buildJsonObject {
            put("id", ids.incrementAndGet())
            put("method", method)
            put("params", params)
        }
        input.appendLine(request.toString()).flush()
        val response = Json.parseToJsonElement(output.readLine()).jsonObject
        if (response["ok"]?.jsonPrimitive?.booleanOrNull != true) {
            error(response["error"]?.jsonPrimitive?.content ?: "Kernelyra error")
        }
        return response.getValue("result")
    }

    fun plan(dataset: String, options: JsonObject = buildJsonObject {}) = call("plan", withPaths(options, dataset))
    fun train(dataset: String, options: JsonObject = buildJsonObject {}) = call("train", withPaths(options, dataset))
    fun finetune(model: String, dataset: String, options: JsonObject = buildJsonObject {}) =
        call("finetune", withPaths(options, dataset, model))

    private fun withPaths(options: JsonObject, dataset: String, model: String? = null) = buildJsonObject {
        options.forEach { (key, value) -> put(key, value) }
        put("dataset", dataset)
        model?.let { put("model", it) }
    }

    override fun close() {
        input.close()
        if (!process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)) process.destroyForcibly()
    }
}
