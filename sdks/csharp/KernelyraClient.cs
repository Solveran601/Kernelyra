using System.Diagnostics;
using System.Text.Json;

namespace Kernelyra;

public sealed class Config
{
    internal Dictionary<string, object?> Values { get; } = new();
    public static Config Auto(string? target = null) => new Config().Target(target);
    public Config Set(string name, object? value) { if (value is not null) Values[name] = value; return this; }
    public Config Target(string? value) => Set("target", value);
    public Config Task(string value) => Set("task", value);
    public Config Backend(string value) => Set("backend", value);
    public Config Architecture(string value) => Set("architecture", value);
    public Config ModelFormat(string value) => Set("model_format", value);
    public Config Profile(string value) => Set("profile", value);
    public Config Goal(double value) => Set("target_metric", value);
    public Config Steps(long value) => Set("max_steps", value);
    public Config Batch(int value, bool acceptRisk = false) => Set("batch_size", value).Set("accept_batch_risk", acceptRisk);
    public Config Resources(int cpu, int ram, int gpu = 0) => Set("cpu", cpu).Set("ram", ram).Set("gpu", gpu);
    public Config Optimizer(double learningRate, double weightDecay = 0) => Set("learning_rate", learningRate).Set("weight_decay", weightDecay);
    public Config Model(string precision = "auto", params int[] hiddenLayers)
    {
        Set("precision", precision);
        if (hiddenLayers.Length > 0) Set("hidden_layers", hiddenLayers);
        return this;
    }
    public Config Data(int workers, int prefetch = 1) => Set("data_workers", workers).Set("prefetch", prefetch);
    public Config Quality(int interval, double minImprovement = .0005, int earlyStoppingPatience = 18, int targetPatience = 3) =>
        Set("evaluation_interval", interval).Set("min_improvement", minImprovement)
            .Set("early_stopping_patience", earlyStoppingPatience).Set("target_patience", targetPatience);
    public Config Guard(double margin = .03, int patience = 3) =>
        Set("degradation_margin", margin).Set("degradation_patience", patience);
    public Config Seed(int value) => Set("seed", value);
}

public sealed record TrainingResult(string? Checkpoint, JsonElement Plan, JsonElement Dataset, JsonElement Run)
{
    public string? Status => Run.TryGetProperty("status", out var value) ? value.GetString() : null;
    public JsonElement Metrics => Run.TryGetProperty("metrics", out var value) ? value : default;
}

public sealed class Client : IDisposable
{
    private readonly Process process;
    private readonly object gate = new();
    private long requestId;
    private bool disposed;

    public Client(string workspace, string executable = "kernelyra")
    {
        if (string.IsNullOrWhiteSpace(executable))
            throw new ArgumentException("Kernelyra executable is required", nameof(executable));
        var start = new ProcessStartInfo(executable) {
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = false,
            CreateNoWindow = true,
        };
        start.ArgumentList.Add("--workspace");
        start.ArgumentList.Add(workspace);
        start.ArgumentList.Add("rpc");
        process = Process.Start(start) ?? throw new InvalidOperationException("Cannot start Kernelyra");
        try {
            var ready = process.StandardOutput.ReadLine() ?? throw new EndOfStreamException("Kernelyra did not start");
            using var payload = JsonDocument.Parse(ready);
            if (payload.RootElement.GetProperty("protocol").GetString() != "kernelyra-jsonl/1")
                throw new InvalidDataException("Incompatible Kernelyra protocol");
        }
        catch {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
            process.Dispose();
            throw;
        }
    }

    public JsonElement Call(string method, object? parameters = null)
    {
        lock (gate) {
            ObjectDisposedException.ThrowIf(disposed, this);
            var id = Interlocked.Increment(ref requestId);
            process.StandardInput.WriteLine(JsonSerializer.Serialize(new { id, method, @params = parameters ?? new { } }));
            var line = process.StandardOutput.ReadLine() ?? throw new EndOfStreamException("Kernelyra closed the protocol");
            using var response = JsonDocument.Parse(line);
            if (!response.RootElement.GetProperty("ok").GetBoolean())
                throw new InvalidOperationException(response.RootElement.GetProperty("error").GetString());
            return response.RootElement.GetProperty("result").Clone();
        }
    }

    public JsonElement Plan(string dataset, object? options = null) =>
        Call("plan", Merge(dataset, null, options));

    public JsonElement Train(string dataset, object? options = null) =>
        Call("train", Merge(dataset, null, options));

    public JsonElement FineTune(string model, string dataset, object? options = null) =>
        Call("finetune", Merge(dataset, model, options));

    public TrainingResult Fit(string dataset, string? target = null, Config? config = null)
    {
        var values = config is null ? new Dictionary<string, object?>() : new(config.Values);
        if (target is not null) values["target"] = target;
        var result = Call("train", Merge(dataset, null, values));
        return JsonSerializer.Deserialize<TrainingResult>(result.GetRawText(), new JsonSerializerOptions {
            PropertyNameCaseInsensitive = true,
        }) ?? throw new InvalidDataException("Invalid Kernelyra result");
    }

    public TrainingResult Tune(string model, string dataset, string? target = null, Config? config = null)
    {
        var values = config is null ? new Dictionary<string, object?>() : new(config.Values);
        if (target is not null) values["target"] = target;
        var result = Call("finetune", Merge(dataset, model, values));
        return JsonSerializer.Deserialize<TrainingResult>(result.GetRawText(), new JsonSerializerOptions {
            PropertyNameCaseInsensitive = true,
        }) ?? throw new InvalidDataException("Invalid Kernelyra result");
    }

    private static Dictionary<string, object?> Merge(string dataset, string? model, object? options)
    {
        var result = options is null
            ? new Dictionary<string, object?>()
            : JsonSerializer.Deserialize<Dictionary<string, object?>>(JsonSerializer.Serialize(options))!;
        result["dataset"] = dataset;
        if (model is not null) result["model"] = model;
        return result;
    }

    public void Dispose()
    {
        lock (gate) {
            if (disposed) return;
            disposed = true;
            process.StandardInput.Close();
            if (!process.WaitForExit(5000)) process.Kill(entireProcessTree: true);
            process.Dispose();
        }
    }
}
