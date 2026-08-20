package main

import (
	"fmt"
	"os"
	"strconv"

	kernelyra "github.com/kernelyra-ai/kernelyra-go"
)

func argument(index int, fallback string) string {
	if len(os.Args) > index {
		return os.Args[index]
	}
	return fallback
}

func main() {
	dataset := argument(1, "train.csv")
	target := argument(2, "label")
	workspace := argument(3, "./workspace")
	executable := argument(4, "kernelyra")
	backend := argument(5, "torch")
	steps, err := strconv.Atoi(argument(6, "5000"))
	if err != nil {
		panic(err)
	}
	engine, err := kernelyra.OpenWithExecutable(workspace, executable)
	if err != nil {
		panic(err)
	}
	defer engine.Close()
	result, err := engine.Fit(dataset, target, kernelyra.Auto("").WithBackend(backend).WithGoal(0.95).WithSteps(steps))
	if err != nil {
		panic(err)
	}
	fmt.Printf("status=%s checkpoint=%s\n", result.Status(), result.Checkpoint)
}
