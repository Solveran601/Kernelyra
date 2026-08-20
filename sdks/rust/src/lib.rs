use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[cfg(feature = "native")]
pub mod native;

pub struct Client {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    id: u64,
}

#[derive(Default, Clone)]
pub struct Config {
    values: Map<String, Value>,
}

impl Config {
    pub fn auto(target: &str) -> Self {
        if target.is_empty() {
            Self::default()
        } else {
            Self::default().target(target)
        }
    }
    pub fn set(mut self, name: &str, value: impl Into<Value>) -> Self {
        self.values.insert(name.into(), value.into());
        self
    }
    pub fn target(self, value: &str) -> Self {
        self.set("target", value)
    }
    pub fn task(self, value: &str) -> Self {
        self.set("task", value)
    }
    pub fn backend(self, value: &str) -> Self {
        self.set("backend", value)
    }
    pub fn architecture(self, value: &str) -> Self {
        self.set("architecture", value)
    }
    pub fn model_format(self, value: &str) -> Self {
        self.set("model_format", value)
    }
    pub fn profile(self, value: &str) -> Self {
        self.set("profile", value)
    }
    pub fn goal(self, value: f64) -> Self {
        self.set("target_metric", value)
    }
    pub fn steps(self, value: u64) -> Self {
        self.set("max_steps", value)
    }
    pub fn batch(mut self, value: u64, accept_risk: bool) -> Self {
        self.values.insert("batch_size".into(), value.into());
        self.values
            .insert("accept_batch_risk".into(), accept_risk.into());
        self
    }
    pub fn resources(mut self, cpu: u64, ram: u64, gpu: u64) -> Self {
        self.values.insert("cpu".into(), cpu.into());
        self.values.insert("ram".into(), ram.into());
        self.values.insert("gpu".into(), gpu.into());
        self
    }
    pub fn optimizer(mut self, learning_rate: f64, weight_decay: f64) -> Self {
        self.values
            .insert("learning_rate".into(), learning_rate.into());
        self.values
            .insert("weight_decay".into(), weight_decay.into());
        self
    }
    pub fn model(mut self, hidden_layers: &[u64], precision: &str) -> Self {
        if !hidden_layers.is_empty() {
            self.values
                .insert("hidden_layers".into(), json!(hidden_layers));
        }
        self.values.insert("precision".into(), precision.into());
        self
    }
    pub fn data(mut self, workers: u64, prefetch: u64) -> Self {
        self.values.insert("data_workers".into(), workers.into());
        self.values.insert("prefetch".into(), prefetch.into());
        self
    }
    pub fn quality(
        mut self,
        interval: u64,
        min_improvement: f64,
        early_stopping_patience: u64,
        target_patience: u64,
    ) -> Self {
        self.values.insert("evaluation_interval".into(), interval.into());
        self.values.insert("min_improvement".into(), min_improvement.into());
        self.values.insert("early_stopping_patience".into(), early_stopping_patience.into());
        self.values.insert("target_patience".into(), target_patience.into());
        self
    }
    pub fn guard(mut self, margin: f64, patience: u64) -> Self {
        self.values.insert("degradation_margin".into(), margin.into());
        self.values.insert("degradation_patience".into(), patience.into());
        self
    }
    pub fn seed(self, value: u64) -> Self {
        self.set("seed", value)
    }
    pub fn into_map(self) -> Map<String, Value> {
        self.values
    }
}

pub struct TrainingResult {
    pub checkpoint: Option<String>,
    pub plan: Value,
    pub dataset: Value,
    pub run: Value,
}

impl TrainingResult {
    fn from_value(value: Value) -> Result<Self, String> {
        Ok(Self {
            checkpoint: value["checkpoint"].as_str().map(str::to_owned),
            plan: value.get("plan").cloned().ok_or("missing plan")?,
            dataset: value.get("dataset").cloned().ok_or("missing dataset")?,
            run: value.get("run").cloned().ok_or("missing run")?,
        })
    }
    pub fn status(&self) -> Option<&str> {
        self.run["status"].as_str()
    }
    pub fn metrics(&self) -> &Value {
        &self.run["metrics"]
    }
}

impl Client {
    pub fn open(workspace: &str) -> Result<Self, String> {
        Self::open_with_executable(workspace, "kernelyra")
    }

    pub fn open_with_executable(workspace: &str, executable: &str) -> Result<Self, String> {
        if executable.is_empty() {
            return Err("Kernelyra executable is required".into());
        }
        let mut child = Command::new(executable)
            .args(["--workspace", workspace, "rpc"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("missing stdin")?;
        let mut output = BufReader::new(child.stdout.take().ok_or("missing stdout")?);
        let mut line = String::new();
        output.read_line(&mut line).map_err(|e| e.to_string())?;
        let ready: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if ready["protocol"] != "kernelyra-jsonl/1" {
            return Err("incompatible protocol".into());
        }
        Ok(Self {
            child,
            input,
            output,
            id: 0,
        })
    }

    /// Easy API: dataset plus target are enough; None means full auto.
    pub fn fit(
        &mut self,
        dataset: &str,
        target: &str,
        config: Option<Config>,
    ) -> Result<TrainingResult, String> {
        let mut options = config.unwrap_or_default().into_map();
        options.insert("dataset".into(), dataset.into());
        if !target.is_empty() {
            options.insert("target".into(), target.into());
        }
        TrainingResult::from_value(self.call("train", Value::Object(options))?)
    }

    pub fn tune(
        &mut self,
        model: &str,
        dataset: &str,
        target: &str,
        config: Option<Config>,
    ) -> Result<TrainingResult, String> {
        let mut options = config.unwrap_or_default().into_map();
        options.insert("model".into(), model.into());
        options.insert("dataset".into(), dataset.into());
        if !target.is_empty() {
            options.insert("target".into(), target.into());
        }
        TrainingResult::from_value(self.call("finetune", Value::Object(options))?)
    }

    pub fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.id += 1;
        writeln!(
            self.input,
            "{}",
            json!({"id":self.id,"method":method,"params":params})
        )
        .map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.output
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        let response: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if response["ok"] != true {
            return Err(response["error"]
                .as_str()
                .unwrap_or("Kernelyra error")
                .into());
        }
        Ok(response["result"].clone())
    }

    pub fn plan(
        &mut self,
        dataset: &str,
        mut options: Map<String, Value>,
    ) -> Result<Value, String> {
        options.insert("dataset".into(), Value::String(dataset.into()));
        self.call("plan", Value::Object(options))
    }

    pub fn train(
        &mut self,
        dataset: &str,
        mut options: Map<String, Value>,
    ) -> Result<Value, String> {
        options.insert("dataset".into(), Value::String(dataset.into()));
        self.call("train", Value::Object(options))
    }

    pub fn finetune(
        &mut self,
        model: &str,
        dataset: &str,
        mut options: Map<String, Value>,
    ) -> Result<Value, String> {
        options.insert("model".into(), Value::String(model.into()));
        options.insert("dataset".into(), Value::String(dataset.into()));
        self.call("finetune", Value::Object(options))
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.input.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn auto_without_target_preserves_inference() {
        assert!(!Config::auto("").into_map().contains_key("target"));
    }

    #[test]
    fn explicit_zero_resources_are_preserved() {
        let values = Config::default().resources(35, 40, 0).data(1, 0).into_map();
        assert_eq!(values["gpu"], 0);
        assert_eq!(values["prefetch"], 0);
    }

    #[test]
    fn model_without_layers_preserves_engine_layers() {
        let values = Config::default().model(&[], "fp32").into_map();
        assert!(!values.contains_key("hidden_layers"));
        assert_eq!(values["precision"], "fp32");
    }
}
