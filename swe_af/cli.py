"""CLI client for SWE-AF.

Provides a command-line interface to trigger builds, check status,
and manage SWE-AF workflows via the AgentField control plane.

Usage:
    swe build "Add JWT auth" --path ./my-project
    swe plan "Refactor API" --path ./my-project
    swe status <execution_id>
    swe resume ./my-project
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import click
import httpx


# Default configuration
DEFAULT_SERVER = os.getenv("AGENTFIELD_SERVER", "http://localhost:8080")
DEFAULT_TIMEOUT = 300.0  # 5 minutes for long operations
POLL_INTERVAL = 2.0  # seconds between status checks


def get_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    """Create an HTTP client for the AgentField control plane."""
    return httpx.Client(timeout=timeout)


def parse_model_option(models: tuple[str, ...]) -> dict[str, str]:
    """Parse --model options into a dict.

    Args:
        models: Tuple of "key=value" strings like ("default=sonnet", "coder=opus")

    Returns:
        Dict mapping role to model ID
    """
    result = {}
    for model_spec in models:
        if "=" not in model_spec:
            raise click.BadParameter(
                f"Model must be in format 'role=model', got: {model_spec}"
            )
        key, value = model_spec.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def build_config(
    runtime: str,
    models: dict[str, str],
    max_coding_iterations: int | None,
    max_advisor_invocations: int | None,
    max_replans: int | None,
    enable_learning: bool,
    permission_mode: str | None,
    max_turns: int | None,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    """Build the config dict for the API request."""
    config: dict[str, Any] = {"runtime": runtime}

    if models:
        config["models"] = models

    if max_coding_iterations is not None:
        config["max_coding_iterations"] = max_coding_iterations
    if max_advisor_invocations is not None:
        config["max_advisor_invocations"] = max_advisor_invocations
    if max_replans is not None:
        config["max_replans"] = max_replans
    if enable_learning:
        config["enable_learning"] = True
    if permission_mode:
        config["permission_mode"] = permission_mode
    if max_turns is not None:
        config["agent_max_turns"] = max_turns
    if timeout_seconds is not None:
        config["agent_timeout_seconds"] = timeout_seconds

    return config


def trigger_build(
    client: httpx.Client,
    goal: str,
    repo_path: str,
    repo_url: str,
    config: dict[str, Any],
    additional_context: str,
    server: str,
) -> str:
    """Trigger an async build and return the execution_id."""
    endpoint = f"{server}/api/v1/execute/async/swe-planner.build"

    payload: dict[str, Any] = {
        "input": {
            "goal": goal,
            "config": config,
        }
    }

    if repo_path:
        payload["input"]["repo_path"] = repo_path
    if repo_url:
        payload["input"]["repo_url"] = repo_url
    if additional_context:
        payload["input"]["additional_context"] = additional_context

    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    result = response.json()

    execution_id = result.get("execution_id")
    if not execution_id:
        raise click.ClickException(
            f"No execution_id in response: {json.dumps(result, indent=2)}"
        )

    return execution_id


def trigger_plan(
    client: httpx.Client,
    goal: str,
    repo_path: str,
    additional_context: str,
    server: str,
    config: dict[str, Any],
) -> str:
    """Trigger an async plan and return the execution_id."""
    endpoint = f"{server}/api/v1/execute/async/swe-planner.plan"

    # Extract model overrides for plan-specific agents
    models = config.get("models", {})

    payload: dict[str, Any] = {
        "input": {
            "goal": goal,
            "repo_path": repo_path,
            "additional_context": additional_context,
            "pm_model": models.get("pm", models.get("default", "sonnet")),
            "architect_model": models.get("architect", models.get("default", "sonnet")),
            "tech_lead_model": models.get("tech_lead", models.get("default", "sonnet")),
            "sprint_planner_model": models.get(
                "sprint_planner", models.get("default", "sonnet")
            ),
            "issue_writer_model": models.get(
                "issue_writer", models.get("default", "sonnet")
            ),
            "permission_mode": config.get("permission_mode", ""),
            "ai_provider": "claude"
            if config.get("runtime") == "claude_code"
            else "open_code",
        }
    }

    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    result = response.json()

    execution_id = result.get("execution_id")
    if not execution_id:
        raise click.ClickException(
            f"No execution_id in response: {json.dumps(result, indent=2)}"
        )

    return execution_id


def get_status(client: httpx.Client, execution_id: str, server: str) -> dict[str, Any]:
    """Get the status of an execution."""
    endpoint = f"{server}/api/v1/executions/{execution_id}"
    response = client.get(endpoint)
    response.raise_for_status()
    return response.json()


def format_status(status: dict[str, Any]) -> str:
    """Format execution status for display."""
    lines = []

    exec_id = status.get("execution_id", "unknown")
    state = status.get("status", "unknown")
    node_id = status.get("node_id", "unknown")
    reasoner = status.get("reasoner", "unknown")

    lines.append(f"Execution: {exec_id}")
    lines.append(f"Status:    {state}")
    lines.append(f"Agent:     {node_id}.{reasoner}")

    if "result" in status and status["result"]:
        result = status["result"]
        if isinstance(result, dict):
            lines.append("")
            lines.append("Result:")
            lines.append(json.dumps(result, indent=2))

    if "error" in status and status["error"]:
        lines.append("")
        lines.append(f"Error: {status['error']}")

    return "\n".join(lines)


def poll_execution(
    client: httpx.Client,
    execution_id: str,
    server: str,
    poll: bool,
    verbose: bool = False,
) -> dict[str, Any]:
    """Poll execution until complete if --poll is set."""
    if not poll:
        return get_status(client, execution_id, server)

    click.echo(f"Polling execution {execution_id}...")
    click.echo("Press Ctrl+C to stop polling.\n")

    last_status = None
    while True:
        try:
            status = get_status(client, execution_id, server)
            state = status.get("status", "unknown")

            # Show status changes
            if state != last_status:
                timestamp = time.strftime("%H:%M:%S")
                click.echo(f"[{timestamp}] Status: {state}")
                last_status = state

            # Check if complete
            if state in ("completed", "failed", "cancelled"):
                click.echo("")
                return status

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            click.echo("\n\nPolling stopped. Execution continues on the server.")
            click.echo(f"Check status with: swe status {execution_id}")
            sys.exit(0)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="0.1.0", prog_name="swe")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER,
    envvar="AGENTFIELD_SERVER",
    help="AgentField control plane URL",
)
@click.pass_context
def cli(ctx: click.Context, server: str) -> None:
    """SWE-AF CLI client for autonomous software engineering.

    Run builds, check status, and manage SWE-AF workflows.

    \b
    Commands:
        build       Trigger a full autonomous build
        plan        Generate plan (PRD + architecture + issues) without execution
        execute     Execute a prebuilt plan from artifacts
        status      Check execution status
        resume      Resume a crashed build from checkpoint
        cancel      Cancel a running execution
        list        List recent executions
        logs        View build logs from artifacts
        verify      Run verification against acceptance criteria
        call        Call a specific agent/reasoner directly
        agents      List registered agents

    \b
    Examples:
        swe build "Add JWT auth" --path ./my-project
        swe build "Refactor API" --path ./my-project --model coder=opus
        swe status abc123 --poll
        swe resume ./my-project

    \b
    Quick Start:
        1. Start the AgentField control plane: af
        2. Start SWE-AF agent: python -m swe_af
        3. Trigger a build: swe build "Your goal" --path ./your-repo

    \b
    Environment:
        AGENTFIELD_SERVER    Control plane URL (default: http://localhost:8080)
    """
    ctx.ensure_object(dict)
    ctx.obj["server"] = server


@cli.command()
@click.argument("goal")
@click.option(
    "--path",
    "-p",
    "repo_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to repository (default: current directory)",
)
@click.option(
    "--url",
    "-u",
    "repo_url",
    default=None,
    help="Git URL to clone (alternative to --path)",
)
@click.option(
    "--runtime",
    "-r",
    type=click.Choice(["claude_code", "open_code"]),
    default="claude_code",
    help="Model runtime to use",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    help="Model mapping in format 'role=model' (e.g., --model default=sonnet --model coder=opus)",
)
@click.option(
    "--max-coding-iterations",
    type=int,
    default=None,
    help="Max inner-loop retries (default: 5)",
)
@click.option(
    "--max-advisor-invocations",
    type=int,
    default=None,
    help="Max middle-loop advisor calls (default: 2)",
)
@click.option(
    "--max-replans",
    type=int,
    default=None,
    help="Max outer-loop replans (default: 2)",
)
@click.option(
    "--learning/--no-learning",
    default=False,
    help="Enable continual learning across issues",
)
@click.option(
    "--permission-mode",
    type=click.Choice(["", "accept-all", "auto"]),
    default=None,
    help="Permission mode for agent actions",
)
@click.option(
    "--max-turns",
    type=int,
    default=None,
    help="Max tool-use turns per agent",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=int,
    default=None,
    help="Per-agent timeout in seconds",
)
@click.option(
    "--context",
    "-c",
    "additional_context",
    default="",
    help="Additional context for the build",
)
@click.option(
    "--poll",
    is_flag=True,
    default=False,
    help="Poll until build completes",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output",
)
@click.pass_context
def build(
    ctx: click.Context,
    goal: str,
    repo_path: str | None,
    repo_url: str | None,
    runtime: str,
    models: tuple[str, ...],
    max_coding_iterations: int | None,
    max_advisor_invocations: int | None,
    max_replans: int | None,
    learning: bool,
    permission_mode: str | None,
    max_turns: int | None,
    timeout_seconds: int | None,
    additional_context: str,
    poll: bool,
    verbose: bool,
) -> None:
    """Trigger an autonomous build.

    GOAL is the natural language description of what to build.

    \b
    Examples:
        swe build "Add JWT auth to all API endpoints"
        swe build "Refactor auth module" --path ./my-project --model coder=opus
        swe build "Add tests" --path ./my-project --learning --poll
    """
    server = ctx.obj["server"]

    # Default to current directory if no path/url specified
    if not repo_path and not repo_url:
        repo_path = os.getcwd()

    # Parse model options
    model_dict = parse_model_option(models) if models else {}

    # Build config
    config = build_config(
        runtime=runtime,
        models=model_dict,
        max_coding_iterations=max_coding_iterations,
        max_advisor_invocations=max_advisor_invocations,
        max_replans=max_replans,
        enable_learning=learning,
        permission_mode=permission_mode,
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
    )

    if verbose:
        click.echo(f"Server: {server}")
        click.echo(f"Goal: {goal}")
        click.echo(f"Path: {repo_path or 'N/A'}")
        click.echo(f"URL: {repo_url or 'N/A'}")
        click.echo(f"Config: {json.dumps(config, indent=2)}")
        click.echo("")

    # Trigger build
    with get_client() as client:
        try:
            execution_id = trigger_build(
                client=client,
                goal=goal,
                repo_path=repo_path or "",
                repo_url=repo_url or "",
                config=config,
                additional_context=additional_context,
                server=server,
            )
            click.echo(f"Build started: {execution_id}")
            click.echo(f"Check status: swe status {execution_id}")

            if poll:
                click.echo("")
                result = poll_execution(
                    client, execution_id, server, poll=True, verbose=verbose
                )
                click.echo(format_status(result))
            else:
                click.echo(f"Poll for updates: swe status {execution_id} --poll")

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}. "
                "Is the control plane running? Start with: af"
            ) from e


@cli.command()
@click.argument("goal")
@click.option(
    "--path",
    "-p",
    "repo_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to repository (default: current directory)",
)
@click.option(
    "--runtime",
    "-r",
    type=click.Choice(["claude_code", "open_code"]),
    default="claude_code",
    help="Model runtime to use",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    help="Model mapping for planning agents (e.g., --model pm=sonnet)",
)
@click.option(
    "--context",
    "-c",
    "additional_context",
    default="",
    help="Additional context for planning",
)
@click.option(
    "--poll",
    is_flag=True,
    default=False,
    help="Poll until plan completes",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed output",
)
@click.pass_context
def plan(
    ctx: click.Context,
    goal: str,
    repo_path: str | None,
    runtime: str,
    models: tuple[str, ...],
    additional_context: str,
    poll: bool,
    verbose: bool,
) -> None:
    """Run planning only (no execution).

    Generates PRD, architecture, and issue DAG without coding.

    \b
    Examples:
        swe plan "Add authentication system"
        swe plan "Refactor API" --path ./my-project --poll
    """
    server = ctx.obj["server"]

    if not repo_path:
        repo_path = os.getcwd()

    model_dict = parse_model_option(models) if models else {}
    config = {"runtime": runtime, "models": model_dict}

    if verbose:
        click.echo(f"Server: {server}")
        click.echo(f"Goal: {goal}")
        click.echo(f"Path: {repo_path}")
        click.echo("")

    with get_client() as client:
        try:
            execution_id = trigger_plan(
                client=client,
                goal=goal,
                repo_path=repo_path or "",
                additional_context=additional_context,
                server=server,
                config=config,
            )
            click.echo(f"Plan started: {execution_id}")

            if poll:
                click.echo("")
                result = poll_execution(
                    client, execution_id, server, poll=True, verbose=verbose
                )
                click.echo(format_status(result))
            else:
                click.echo(f"Check status: swe status {execution_id}")

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}. "
                "Is the control plane running?"
            ) from e


@cli.command()
@click.argument("execution_id")
@click.option(
    "--poll",
    is_flag=True,
    default=False,
    help="Poll until execution completes",
)
@click.option(
    "--json-output",
    "-j",
    "json_output",
    is_flag=True,
    default=False,
    help="Output raw JSON",
)
@click.pass_context
def status(
    ctx: click.Context,
    execution_id: str,
    poll: bool,
    json_output: bool,
) -> None:
    """Check execution status.

    \b
    Examples:
        swe status abc123
        swe status abc123 --poll
        swe status abc123 --json
    """
    server = ctx.obj["server"]

    with get_client() as client:
        try:
            if poll:
                result = poll_execution(client, execution_id, server, poll=True)
            else:
                result = get_status(client, execution_id, server)

            if json_output:
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(format_status(result))

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option(
    "--artifacts-dir",
    default=".artifacts",
    help="Artifacts directory path",
)
@click.option(
    "--poll",
    is_flag=True,
    default=False,
    help="Poll until resume completes",
)
@click.pass_context
def resume(
    ctx: click.Context,
    repo_path: str,
    artifacts_dir: str,
    poll: bool,
) -> None:
    """Resume a crashed build from checkpoint.

    \b
    Examples:
        swe resume ./my-project
        swe resume ./my-project --poll
    """
    server = ctx.obj["server"]
    endpoint = f"{server}/api/v1/execute/async/swe-planner.resume_build"

    payload = {
        "input": {
            "repo_path": os.path.abspath(repo_path),
            "artifacts_dir": artifacts_dir,
        }
    }

    with get_client() as client:
        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            execution_id = result.get("execution_id")

            click.echo(f"Resume started: {execution_id}")

            if poll:
                click.echo("")
                result = poll_execution(client, execution_id, server, poll=True)
                click.echo(format_status(result))
            else:
                click.echo(f"Check status: swe status {execution_id}")

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


@cli.command("list")
@click.option(
    "--limit",
    "-n",
    default=10,
    help="Maximum number of executions to show",
)
@click.pass_context
def list_executions(ctx: click.Context, limit: int) -> None:
    """List recent executions.

    \b
    Examples:
        swe list
        swe list --limit 20
    """
    server = ctx.obj["server"]
    endpoint = f"{server}/api/v1/executions"

    with get_client() as client:
        try:
            response = client.get(endpoint, params={"limit": limit})
            response.raise_for_status()
            result = response.json()

            executions = result.get("executions", [])
            if not executions:
                click.echo("No executions found.")
                return

            click.echo(f"Recent executions (showing {len(executions)}):\n")
            for exec_data in executions:
                exec_id = exec_data.get("execution_id", "unknown")
                status_val = exec_data.get("status", "unknown")
                reasoner = exec_data.get("reasoner", "unknown")
                created = exec_data.get("created_at", "unknown")
                click.echo(
                    f"  {exec_id[:12]}...  {status_val:12}  {reasoner:20}  {created}"
                )

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


@cli.command()
@click.pass_context
def agents(ctx: click.Context) -> None:
    """List available agents and reasoners.

    \b
    Examples:
        swe agents
    """
    server = ctx.obj["server"]

    # Try to get agent info
    with get_client() as client:
        try:
            # Try the nodes endpoint
            response = client.get(f"{server}/api/v1/nodes")
            response.raise_for_status()
            result = response.json()

            nodes = result.get("nodes", [])
            if not nodes:
                click.echo("No agents registered.")
                return

            click.echo("Registered agents:\n")
            for node in nodes:
                node_id = node.get("node_id", "unknown")
                version = node.get("version", "unknown")
                description = node.get("description", "")
                reasoners = node.get("reasoners", [])

                click.echo(f"  {node_id} (v{version})")
                if description:
                    click.echo(f"    {description}")
                if reasoners:
                    click.echo(f"    Reasoners: {', '.join(reasoners)}")
                click.echo()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                click.echo("Agent listing not available on this server version.")
            else:
                raise click.ClickException(
                    f"HTTP {e.response.status_code}: {e.response.text}"
                ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option(
    "--artifacts-dir",
    default=".artifacts",
    help="Artifacts directory containing plan",
)
@click.option(
    "--runtime",
    "-r",
    type=click.Choice(["claude_code", "open_code"]),
    default="claude_code",
    help="Model runtime to use",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    help="Model mapping for execution agents",
)
@click.option(
    "--learning/--no-learning",
    default=False,
    help="Enable continual learning across issues",
)
@click.option(
    "--poll",
    is_flag=True,
    default=False,
    help="Poll until execution completes",
)
@click.pass_context
def execute(
    ctx: click.Context,
    repo_path: str,
    artifacts_dir: str,
    runtime: str,
    models: tuple[str, ...],
    learning: bool,
    poll: bool,
) -> None:
    """Execute a prebuilt plan from artifacts.

    Runs the DAG execution phase on an existing plan (from `swe plan`).

    \b
    Examples:
        swe execute ./my-project
        swe execute ./my-project --model coder=opus --poll
    """
    server = ctx.obj["server"]
    endpoint = f"{server}/api/v1/execute/async/swe-planner.execute"

    model_dict = parse_model_option(models) if models else {}
    config = build_config(
        runtime=runtime,
        models=model_dict,
        max_coding_iterations=None,
        max_advisor_invocations=None,
        max_replans=None,
        enable_learning=learning,
        permission_mode=None,
        max_turns=None,
        timeout_seconds=None,
    )

    plan_path = os.path.join(repo_path, artifacts_dir, "plan")
    if not os.path.isdir(plan_path):
        raise click.ClickException(
            f"No plan found at {plan_path}. Run `swe plan` first."
        )

    payload = {
        "input": {
            "repo_path": os.path.abspath(repo_path),
            "artifacts_dir": artifacts_dir,
            "config": config,
        }
    }

    with get_client() as client:
        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            execution_id = result.get("execution_id")

            click.echo(f"Execution started: {execution_id}")

            if poll:
                click.echo("")
                result = poll_execution(client, execution_id, server, poll=True)
                click.echo(format_status(result))
            else:
                click.echo(f"Check status: swe status {execution_id}")

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


@cli.command()
@click.argument("execution_id")
@click.option(
    "--reason",
    default="Cancelled by user",
    help="Reason for cancellation",
)
@click.pass_context
def cancel(ctx: click.Context, execution_id: str, reason: str) -> None:
    """Cancel a running execution.

    \b
    Examples:
        swe cancel abc123
        swe cancel abc123 --reason "No longer needed"
    """
    server = ctx.obj["server"]
    endpoint = f"{server}/api/v1/executions/{execution_id}/cancel"

    with get_client() as client:
        try:
            response = client.post(endpoint, json={"reason": reason})
            response.raise_for_status()
            result = response.json()

            click.echo(f"Execution {execution_id} cancelled.")
            if result.get("status"):
                click.echo(f"Final status: {result['status']}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise click.ClickException(f"Execution {execution_id} not found.")
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


@cli.command()
@click.argument("agent")
@click.argument("params", required=False)
@click.option(
    "--param",
    "-p",
    "extra_params",
    multiple=True,
    help="Additional parameters as key=value",
)
@click.option(
    "--poll",
    is_flag=True,
    default=False,
    help="Poll until execution completes",
)
@click.option(
    "--json-output",
    "-j",
    "json_output",
    is_flag=True,
    default=False,
    help="Output raw JSON",
)
@click.pass_context
def call(
    ctx: click.Context,
    agent: str,
    params: str | None,
    extra_params: tuple[str, ...],
    poll: bool,
    json_output: bool,
) -> None:
    """Call a specific agent/reasoner directly.

    AGENT is the reasoner name (e.g., run_coder, run_qa).
    PARAMS is optional JSON string for input parameters.

    \b
    Examples:
        swe call run_coder '{"issue": {...}, "repo_path": "/path/to/repo"}'
        swe call run_verifier '{"prd": {...}, "repo_path": "/path/to/repo"}' --poll
        swe call run_coder --param repo_path=./my-project
    """
    server = ctx.obj["server"]
    endpoint = f"{server}/api/v1/execute/async/swe-planner.{agent}"

    input_data: dict[str, Any] = {}

    if params:
        try:
            input_data = json.loads(params)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON: {e}") from e

    for param in extra_params:
        if "=" not in param:
            raise click.BadParameter(f"Parameter must be key=value: {param}")
        key, value = param.split("=", 1)
        try:
            input_data[key] = json.loads(value)
        except json.JSONDecodeError:
            input_data[key] = value

    payload = {"input": input_data}

    with get_client() as client:
        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            execution_id = result.get("execution_id")

            click.echo(f"Called swe-planner.{agent}")
            click.echo(f"Execution: {execution_id}")

            if poll:
                click.echo("")
                result = poll_execution(client, execution_id, server, poll=True)
                if json_output:
                    click.echo(json.dumps(result, indent=2))
                else:
                    click.echo(format_status(result))
            else:
                click.echo(f"Check status: swe status {execution_id}")

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option(
    "--artifacts-dir",
    default=".artifacts",
    help="Artifacts directory path",
)
@click.pass_context
def logs(ctx: click.Context, repo_path: str, artifacts_dir: str) -> None:
    """Show execution logs from a build.

    \b
    Examples:
        swe logs ./my-project
        swe logs ./my-project --artifacts-dir .artifacts
    """
    execution_dir = os.path.join(repo_path, artifacts_dir, "execution")

    if not os.path.isdir(execution_dir):
        raise click.ClickException(
            f"No execution logs found at {execution_dir}. Has a build been run?"
        )

    checkpoint_path = os.path.join(execution_dir, "checkpoint.json")
    if os.path.isfile(checkpoint_path):
        click.echo("=== Build Checkpoint ===\n")
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        click.echo(json.dumps(checkpoint, indent=2))
        click.echo("")

    issue_logs = [
        f
        for f in os.listdir(execution_dir)
        if f.startswith("issue-") and f.endswith(".json")
    ]
    if issue_logs:
        click.echo(f"=== Issue Logs ({len(issue_logs)}) ===\n")
        for log_file in sorted(issue_logs):
            log_path = os.path.join(execution_dir, log_file)
            with open(log_path, "r") as f:
                log_data = json.load(f)
            issue_name = log_data.get("issue_name", log_file)
            status = log_data.get("status", "unknown")
            click.echo(f"  {issue_name}: {status}")

    if not os.path.isfile(checkpoint_path) and not issue_logs:
        click.echo("No logs found.")


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option(
    "--artifacts-dir",
    default=".artifacts",
    help="Artifacts directory path",
)
@click.pass_context
def verify(ctx: click.Context, repo_path: str, artifacts_dir: str) -> None:
    """Run verification against acceptance criteria.

    \b
    Examples:
        swe verify ./my-project
    """
    server = ctx.obj["server"]

    prd_path = os.path.join(repo_path, artifacts_dir, "plan", "prd.md")
    if not os.path.isfile(prd_path):
        raise click.ClickException(f"No PRD found at {prd_path}. Run `swe plan` first.")

    endpoint = f"{server}/api/v1/execute/async/swe-planner.run_verifier"

    payload = {
        "input": {
            "repo_path": os.path.abspath(repo_path),
            "artifacts_dir": artifacts_dir,
            "prd": {},
            "completed_issues": [],
            "failed_issues": [],
            "skipped_issues": [],
        }
    }

    with get_client() as client:
        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            execution_id = result.get("execution_id")

            click.echo(f"Verification started: {execution_id}")
            click.echo(f"Check status: swe status {execution_id} --poll")

        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise click.ClickException(
                f"Cannot connect to AgentField at {server}"
            ) from e


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
