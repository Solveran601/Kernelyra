import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicLong;

public final class KernelyraClient implements AutoCloseable {
    private final Process process;
    private final BufferedWriter input;
    private final BufferedReader output;
    private final AtomicLong id = new AtomicLong();

    public KernelyraClient(String workspace) throws IOException {
        process = new ProcessBuilder("kernelyra", "--workspace", workspace, "rpc").redirectError(ProcessBuilder.Redirect.INHERIT).start();
        input = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
        output = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
        String ready = output.readLine();
        if (ready == null || !ready.contains("\"protocol\":\"kernelyra-jsonl/1\"")) throw new IOException("Incompatible Kernelyra protocol");
    }

    public synchronized String call(String method, String paramsJson) throws IOException {
        long requestId = id.incrementAndGet();
        input.write("{\"id\":" + requestId + ",\"method\":\"" + escape(method) + "\",\"params\":" + paramsJson + "}\n");
        input.flush();
        String response = output.readLine();
        if (response == null) throw new EOFException("Kernelyra closed the protocol");
        if (!response.contains("\"ok\":true")) throw new IOException("Kernelyra request failed: " + response);
        return response;
    }

    public String plan(String dataset) throws IOException { return call("plan", "{\"dataset\":\"" + escape(dataset) + "\"}"); }
    public String train(String dataset) throws IOException { return call("train", "{\"dataset\":\"" + escape(dataset) + "\"}"); }
    public String finetune(String model, String dataset) throws IOException {
        return call("finetune", "{\"model\":\"" + escape(model) + "\",\"dataset\":\"" + escape(dataset) + "\"}");
    }
    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t");
    }
    public void close() throws IOException { input.close(); try { process.waitFor(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); } }
}
