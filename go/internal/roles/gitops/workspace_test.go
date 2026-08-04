package gitops

import "testing"

func TestToMapReturnsTaggedMap(t *testing.T) {
	type sample struct {
		Mode    string `json:"mode"`
		Success bool   `json:"success"`
	}
	s := sample{Mode: "gitflow", Success: true}
	m, err := toMap(&s)
	if err != nil {
		t.Fatalf("toMap error: %v", err)
	}
	if got, want := m["mode"], "gitflow"; got != want {
		t.Fatalf("mode: got %v, want %v", got, want)
	}
	if got, want := m["success"], true; got != want {
		t.Fatalf("success: got %v, want %v", got, want)
	}
