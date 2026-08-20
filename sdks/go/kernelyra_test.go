package kernelyra

import "testing"

func TestAutoWithoutTargetKeepsTargetAutomatic(t *testing.T) {
	params, err := Auto("").params()
	if err != nil {
		t.Fatal(err)
	}
	if _, configured := params["target"]; configured {
		t.Fatal("empty automatic target must not override engine inference")
	}
}

func TestExplicitZeroResourceValuesArePreserved(t *testing.T) {
	params, err := Auto("").WithResources(35, 40, 0).WithData(1, 0).params()
	if err != nil {
		t.Fatal(err)
	}
	if params["gpu"] != 0 || params["prefetch"] != 0 {
		t.Fatalf("explicit zero values were lost: %#v", params)
	}
}

func TestModelWithoutLayersOnlyChangesPrecision(t *testing.T) {
	params, err := Auto("").WithModel("fp32").params()
	if err != nil {
		t.Fatal(err)
	}
	if _, configured := params["hidden_layers"]; configured {
		t.Fatal("empty hidden layers must preserve engine defaults")
	}
}
