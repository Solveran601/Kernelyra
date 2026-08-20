using Kernelyra;

var dataset = args.ElementAtOrDefault(0) ?? "train.csv";
var target = args.ElementAtOrDefault(1) ?? "label";
var workspace = args.ElementAtOrDefault(2) ?? "./workspace";
var executable = args.ElementAtOrDefault(3) ?? "kernelyra";
var backend = args.ElementAtOrDefault(4) ?? "torch";
var steps = long.TryParse(args.ElementAtOrDefault(5), out var parsed) ? parsed : 5_000;
using var engine = new Client(workspace, executable);
var result = engine.Fit(
    dataset,
    target,
    Config.Auto().Backend(backend).Goal(0.95).Steps(steps)
);
Console.WriteLine($"status={result.Status} checkpoint={result.Checkpoint}");
