use kernelyra_client::{Client, Config};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dataset = args.get(1).map(String::as_str).unwrap_or("train.csv");
    let target = args.get(2).map(String::as_str).unwrap_or("label");
    let workspace = args.get(3).map(String::as_str).unwrap_or("./workspace");
    let executable = args.get(4).map(String::as_str).unwrap_or("kernelyra");
    let backend = args.get(5).map(String::as_str).unwrap_or("torch");
    let steps = args
        .get(6)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5_000);
    let mut engine = Client::open_with_executable(workspace, executable)?;
    let result = engine.fit(
        dataset,
        target,
        Some(Config::default().backend(backend).goal(0.95).steps(steps)),
    )?;
    println!(
        "status={} checkpoint={}",
        result.status().unwrap_or("unknown"),
        result.checkpoint.as_deref().unwrap_or("")
    );
    Ok(())
}
