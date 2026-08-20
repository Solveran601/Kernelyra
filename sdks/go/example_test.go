package kernelyra_test

import (
	"fmt"
	kernelyra "github.com/kernelyra-ai/kernelyra-go"
)

func ExampleClient_Fit() {
	client, err := kernelyra.Open("./workspace")
	if err != nil {
		return
	}
	defer client.Close()
	result, err := client.Fit("train.csv", "label", nil)
	if err != nil {
		return
	}
	fmt.Println(result.Checkpoint)
}
