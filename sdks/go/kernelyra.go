package kernelyra

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
)

type Client struct {
	cmd    *exec.Cmd
	in     io.WriteCloser
	out    *bufio.Reader
	mu     sync.Mutex
	id     uint64
	closed bool
}

type response struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result"`
	Error  string          `json:"error"`
}

// Config mirrors the stable Kernelyra engine options. Zero values mean auto.
// Extra keeps forward compatibility with new engine options.
type Config struct {
	Target          string         `json:"target,omitempty"`
	Task            string         `json:"task,omitempty"`
	Backend         string         `json:"backend,omitempty"`
	Architecture    string         `json:"architecture,omitempty"`
	ModelFormat     string         `json:"model_format,omitempty"`
	Profile         string         `json:"profile,omitempty"`
	BatchSize       int            `json:"batch_size,omitempty"`
	AcceptBatchRisk bool           `json:"accept_batch_risk,omitempty"`
	MaxSteps        int            `json:"max_steps,omitempty"`
	TargetMetric    float64        `json:"target_metric,omitempty"`
	CPU             int            `json:"cpu,omitempty"`
	RAM             int            `json:"ram,omitempty"`
	GPU             int            `json:"gpu,omitempty"`
	Seed            int            `json:"seed,omitempty"`
	LearningRate    float64        `json:"learning_rate,omitempty"`
	WeightDecay     float64        `json:"weight_decay,omitempty"`
	HiddenLayers    []int          `json:"hidden_layers,omitempty"`
	Precision       string         `json:"precision,omitempty"`
	DataWorkers     int            `json:"data_workers,omitempty"`
	Prefetch        int            `json:"prefetch,omitempty"`
	EvaluationInterval int         `json:"evaluation_interval,omitempty"`
	MinImprovement float64         `json:"min_improvement,omitempty"`
	DegradationMargin float64      `json:"degradation_margin,omitempty"`
	DegradationPatience int        `json:"degradation_patience,omitempty"`
	EarlyStoppingPatience int      `json:"early_stopping_patience,omitempty"`
	TargetPatience int             `json:"target_patience,omitempty"`
	Extra           map[string]any `json:"-"`
	configured      map[string]bool
}

func Auto(target string) *Config {
	config := &Config{}
	if target != "" {
		config.WithTarget(target)
	}
	return config
}
func (c *Config) mark(names ...string) {
	if c.configured == nil {
		c.configured = map[string]bool{}
	}
	for _, name := range names {
		c.configured[name] = true
	}
}
func (c *Config) WithTarget(value string) *Config  { c.Target = value; c.mark("target"); return c }
func (c *Config) WithTask(value string) *Config    { c.Task = value; c.mark("task"); return c }
func (c *Config) WithBackend(value string) *Config { c.Backend = value; c.mark("backend"); return c }
func (c *Config) WithArchitecture(value string) *Config { c.Architecture = value; c.mark("architecture"); return c }
func (c *Config) WithModelFormat(value string) *Config { c.ModelFormat = value; c.mark("model_format"); return c }
func (c *Config) WithProfile(value string) *Config { c.Profile = value; c.mark("profile"); return c }
func (c *Config) WithGoal(value float64) *Config {
	c.TargetMetric = value
	c.mark("target_metric")
	return c
}
func (c *Config) WithSteps(value int) *Config { c.MaxSteps = value; c.mark("max_steps"); return c }
func (c *Config) WithBatch(value int, acceptRisk bool) *Config {
	c.BatchSize, c.AcceptBatchRisk = value, acceptRisk
	c.mark("batch_size", "accept_batch_risk")
	return c
}
func (c *Config) WithResources(cpu, ram, gpu int) *Config {
	c.CPU, c.RAM, c.GPU = cpu, ram, gpu
	c.mark("cpu", "ram", "gpu")
	return c
}
func (c *Config) WithOptimizer(learningRate, weightDecay float64) *Config {
	c.LearningRate, c.WeightDecay = learningRate, weightDecay
	c.mark("learning_rate", "weight_decay")
	return c
}
func (c *Config) WithModel(precision string, hiddenLayers ...int) *Config {
	c.Precision, c.HiddenLayers = precision, hiddenLayers
	c.mark("precision")
	if len(hiddenLayers) > 0 {
		c.mark("hidden_layers")
	}
	return c
}
func (c *Config) WithData(workers, prefetch int) *Config {
	c.DataWorkers, c.Prefetch = workers, prefetch
	c.mark("data_workers", "prefetch")
	return c
}
func (c *Config) WithQuality(interval int, minImprovement float64, earlyStoppingPatience, targetPatience int) *Config {
	c.EvaluationInterval, c.MinImprovement = interval, minImprovement
	c.EarlyStoppingPatience, c.TargetPatience = earlyStoppingPatience, targetPatience
	c.mark("evaluation_interval", "min_improvement", "early_stopping_patience", "target_patience")
	return c
}
func (c *Config) WithGuard(margin float64, patience int) *Config {
	c.DegradationMargin, c.DegradationPatience = margin, patience
	c.mark("degradation_margin", "degradation_patience")
	return c
}
func (c *Config) WithSeed(value int) *Config { c.Seed = value; c.mark("seed"); return c }
func (c *Config) Set(name string, value any) *Config {
	if c.Extra == nil {
		c.Extra = map[string]any{}
	}
	c.Extra[name] = value
	return c
}

func (c *Config) params() (map[string]any, error) {
	if c == nil {
		return map[string]any{}, nil
	}
	data, err := json.Marshal(c)
	if err != nil {
		return nil, err
	}
	var result map[string]any
	if err = json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	values := map[string]any{
		"target": c.Target, "task": c.Task, "backend": c.Backend, "architecture": c.Architecture,
		"model_format": c.ModelFormat, "profile": c.Profile,
		"batch_size": c.BatchSize, "accept_batch_risk": c.AcceptBatchRisk,
		"max_steps": c.MaxSteps, "target_metric": c.TargetMetric,
		"cpu": c.CPU, "ram": c.RAM, "gpu": c.GPU, "seed": c.Seed,
		"learning_rate": c.LearningRate, "weight_decay": c.WeightDecay,
		"hidden_layers": c.HiddenLayers, "precision": c.Precision,
		"data_workers": c.DataWorkers, "prefetch": c.Prefetch,
		"evaluation_interval": c.EvaluationInterval, "min_improvement": c.MinImprovement,
		"degradation_margin": c.DegradationMargin, "degradation_patience": c.DegradationPatience,
		"early_stopping_patience": c.EarlyStoppingPatience, "target_patience": c.TargetPatience,
	}
	for name := range c.configured {
		result[name] = values[name]
	}
	for key, value := range c.Extra {
		result[key] = value
	}
	return result, nil
}

type TrainingResult struct {
	Checkpoint string          `json:"checkpoint"`
	Plan       json.RawMessage `json:"plan"`
	Dataset    json.RawMessage `json:"dataset"`
	Run        json.RawMessage `json:"run"`
}

func (r *TrainingResult) Status() string {
	var run struct {
		Status string `json:"status"`
	}
	_ = json.Unmarshal(r.Run, &run)
	return run.Status
}

func (r *TrainingResult) Metrics() json.RawMessage {
	var run struct {
		Metrics json.RawMessage `json:"metrics"`
	}
	_ = json.Unmarshal(r.Run, &run)
	return run.Metrics
}

func Open(workspace string) (*Client, error) {
	return OpenWithExecutable(workspace, "kernelyra")
}

func OpenWithExecutable(workspace, executable string) (*Client, error) {
	if executable == "" {
		return nil, fmt.Errorf("Kernelyra executable is required")
	}
	cmd := exec.Command(executable, "--workspace", workspace, "rpc")
	cmd.Stderr = os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		return nil, err
	}
	c := &Client{cmd: cmd, in: in, out: bufio.NewReader(outPipe)}
	line, err := c.out.ReadBytes('\n')
	if err != nil {
		_ = in.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, err
	}
	var ready map[string]any
	if json.Unmarshal(line, &ready) != nil || ready["protocol"] != "kernelyra-jsonl/1" {
		_ = in.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("incompatible Kernelyra protocol")
	}
	return c, nil
}

// Fit is the easy API: dataset plus target are enough; nil config means full auto.
func (c *Client) Fit(dataset, target string, config *Config) (*TrainingResult, error) {
	params, err := config.params()
	if err != nil {
		return nil, err
	}
	params["dataset"] = dataset
	if target != "" {
		params["target"] = target
	}
	raw, err := c.Call("train", params)
	if err != nil {
		return nil, err
	}
	var result TrainingResult
	if err = json.Unmarshal(raw, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) Tune(model, dataset, target string, config *Config) (*TrainingResult, error) {
	params, err := config.params()
	if err != nil {
		return nil, err
	}
	params["model"], params["dataset"] = model, dataset
	if target != "" {
		params["target"] = target
	}
	raw, err := c.Call("finetune", params)
	if err != nil {
		return nil, err
	}
	var result TrainingResult
	if err = json.Unmarshal(raw, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *Client) Call(method string, params map[string]any) (json.RawMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, fmt.Errorf("Kernelyra client is closed")
	}
	c.id++
	request := map[string]any{"id": c.id, "method": method, "params": params}
	data, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	if _, err = c.in.Write(append(data, '\n')); err != nil {
		return nil, err
	}
	line, err := c.out.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var value response
	if err = json.Unmarshal(line, &value); err != nil {
		return nil, err
	}
	if !value.OK {
		return nil, fmt.Errorf("kernelyra: %s", value.Error)
	}
	return value.Result, nil
}

func with(options map[string]any, values map[string]any) map[string]any {
	result := map[string]any{}
	for k, v := range options {
		result[k] = v
	}
	for k, v := range values {
		result[k] = v
	}
	return result
}
func (c *Client) Plan(dataset string, options map[string]any) (json.RawMessage, error) {
	return c.Call("plan", with(options, map[string]any{"dataset": dataset}))
}
func (c *Client) Train(dataset string, options map[string]any) (json.RawMessage, error) {
	return c.Call("train", with(options, map[string]any{"dataset": dataset}))
}
func (c *Client) FineTune(model, dataset string, options map[string]any) (json.RawMessage, error) {
	return c.Call("finetune", with(options, map[string]any{"model": model, "dataset": dataset}))
}
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	_ = c.in.Close()
	return c.cmd.Wait()
}
