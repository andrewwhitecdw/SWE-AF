# SWE-AF CLI

Command-line interface for triggering builds, checking status, and managing SWE-AF workflows via the AgentField control plane.

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
