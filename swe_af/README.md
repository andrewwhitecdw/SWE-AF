# SWE-AF CLI

Command-line interface for triggering builds, checking status, and managing SWE-AF workflows via the AgentField control plane.

## Quick Start

**Terminal 1 - Start AgentField control plane:**
```bash
cd ~/code/agentfield/control-plane && go run ./cmd/agentfield-server
```

**Terminal 2 - Start SWE-AF agent:**
```bash
cd ~/code/SWE-AF && swe-af
```

**Terminal 3 - Use the CLI:**
```bash
swe agents                    # Verify swe-planner is registered
swe build "Add feature" --path ./my-project --poll
```

## Installation

```bash
cd ~/code/SWE-AF
pipx install -e ".[dev]"
```

## Usage

```bash
swe --help
swe build "Add feature" --path ./my-project
swe status <execution_id> --poll
```

## Running Tests

```bash
cd ~/code/SWE-AF
~/.local/share/pipx/venvs/swe-af/bin/python -m pytest tests/test_cli.py -v
```

Or if you have pytest installed globally:

```bash
cd ~/code/SWE-AF
pytest tests/test_cli.py -v
```

## Commands

| Command | Description |
|---------|-------------|
| `swe build` | Trigger full autonomous build |
| `swe plan` | Generate plan without execution |
| `swe execute` | Execute a prebuilt plan |
| `swe status` | Check execution status |
| `swe resume` | Resume crashed build |
| `swe cancel` | Cancel running execution |
| `swe list` | List recent executions |
| `swe logs` | View build logs |
| `swe verify` | Run verification |
| `swe call` | Call specific agent |
| `swe agents` | List registered agents |
