"""Tests for swe_af.cli module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import click
import httpx
import pytest
from click.testing import CliRunner

from swe_af import cli


class TestParseModelOption:
    """Tests for parse_model_option helper function."""

    def test_empty_tuple_returns_empty_dict(self) -> None:
        result = cli.parse_model_option(())
        assert result == {}

    def test_single_model_mapping(self) -> None:
        result = cli.parse_model_option(("coder=opus",))
        assert result == {"coder": "opus"}

    def test_multiple_model_mappings(self) -> None:
        result = cli.parse_model_option(("default=sonnet", "coder=opus", "qa=haiku"))
        assert result == {"default": "sonnet", "coder": "opus", "qa": "haiku"}

    def test_model_with_spaces_is_stripped(self) -> None:
        result = cli.parse_model_option((" coder = opus ",))
        assert result == {"coder": "opus"}

    def test_model_with_equals_in_value(self) -> None:
        result = cli.parse_model_option(("model=gpt-4=turbo",))
        assert result == {"model": "gpt-4=turbo"}

    def test_invalid_format_raises_bad_parameter(self) -> None:
        with pytest.raises(click.BadParameter, match="Model must be in format"):
            cli.parse_model_option(("invalid_model_spec",))

    def test_empty_string_raises_bad_parameter(self) -> None:
        with pytest.raises(click.BadParameter):
            cli.parse_model_option(("",))


class TestBuildConfig:
    """Tests for build_config helper function."""

    def test_minimal_config_only_runtime(self) -> None:
        result = cli.build_config(
            runtime="claude_code",
            models={},
            max_coding_iterations=None,
            max_advisor_invocations=None,
            max_replans=None,
            enable_learning=False,
            permission_mode=None,
            max_turns=None,
            timeout_seconds=None,
        )
        assert result == {"runtime": "claude_code"}

    def test_config_with_models(self) -> None:
        result = cli.build_config(
            runtime="open_code",
            models={"coder": "opus", "qa": "sonnet"},
            max_coding_iterations=None,
            max_advisor_invocations=None,
            max_replans=None,
            enable_learning=False,
            permission_mode=None,
            max_turns=None,
            timeout_seconds=None,
        )
        assert result == {
            "runtime": "open_code",
            "models": {"coder": "opus", "qa": "sonnet"},
        }

    def test_config_with_all_options(self) -> None:
        result = cli.build_config(
            runtime="claude_code",
            models={"default": "sonnet"},
            max_coding_iterations=10,
            max_advisor_invocations=3,
            max_replans=5,
            enable_learning=True,
            permission_mode="accept-all",
            max_turns=100,
            timeout_seconds=300,
        )
        assert result == {
            "runtime": "claude_code",
            "models": {"default": "sonnet"},
            "max_coding_iterations": 10,
            "max_advisor_invocations": 3,
            "max_replans": 5,
            "enable_learning": True,
            "permission_mode": "accept-all",
            "agent_max_turns": 100,
            "agent_timeout_seconds": 300,
        }

    def test_enable_learning_false_not_included(self) -> None:
        result = cli.build_config(
            runtime="claude_code",
            models={},
            max_coding_iterations=None,
            max_advisor_invocations=None,
            max_replans=None,
            enable_learning=False,
            permission_mode=None,
            max_turns=None,
            timeout_seconds=None,
        )
        assert "enable_learning" not in result

    def test_empty_permission_mode_not_included(self) -> None:
        result = cli.build_config(
            runtime="claude_code",
            models={},
            max_coding_iterations=None,
            max_advisor_invocations=None,
            max_replans=None,
            enable_learning=False,
            permission_mode="",
            max_turns=None,
            timeout_seconds=None,
        )
        assert "permission_mode" not in result


class TestFormatStatus:
    """Tests for format_status helper function."""

    def test_minimal_status(self) -> None:
        result = cli.format_status(
            {
                "execution_id": "abc123",
                "status": "running",
                "node_id": "swe-planner",
                "reasoner": "build",
            }
        )
        assert "Execution: abc123" in result
        assert "Status:    running" in result
        assert "Agent:     swe-planner.build" in result

    def test_status_with_result_dict(self) -> None:
        result = cli.format_status(
            {
                "execution_id": "abc123",
                "status": "completed",
                "node_id": "swe-planner",
                "reasoner": "build",
                "result": {"success": True, "issues_completed": 5},
            }
        )
        assert "Result:" in result
        assert '"success": true' in result
        assert '"issues_completed": 5' in result

    def test_status_with_error(self) -> None:
        result = cli.format_status(
            {
                "execution_id": "abc123",
                "status": "failed",
                "node_id": "swe-planner",
                "reasoner": "build",
                "error": "Connection timeout",
            }
        )
        assert "Error: Connection timeout" in result

    def test_status_with_missing_fields(self) -> None:
        result = cli.format_status({})
        assert "Execution: unknown" in result
        assert "Status:    unknown" in result

    def test_status_with_empty_result(self) -> None:
        result = cli.format_status(
            {
                "execution_id": "abc123",
                "status": "completed",
                "node_id": "swe-planner",
                "reasoner": "build",
                "result": None,
            }
        )
        assert "Result:" not in result

    def test_status_with_non_dict_result(self) -> None:
        result = cli.format_status(
            {
                "execution_id": "abc123",
                "status": "completed",
                "node_id": "swe-planner",
                "reasoner": "build",
                "result": "just a string",
            }
        )
        assert "Result:" not in result


class TestGetStatus:
    """Tests for get_status HTTP function."""

    def test_get_status_success(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "execution_id": "abc123",
            "status": "completed",
        }
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.get.return_value = mock_response

        result = cli.get_status(mock_client, "abc123", "http://localhost:8080")

        mock_client.get.assert_called_once_with(
            "http://localhost:8080/api/v1/executions/abc123"
        )
        assert result == {"execution_id": "abc123", "status": "completed"}

    def test_get_status_not_found(self) -> None:
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=mock.Mock(), response=mock.Mock(status_code=404)
        )

        mock_client = mock.Mock()
        mock_client.get.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            cli.get_status(mock_client, "nonexistent", "http://localhost:8080")


class TestTriggerBuild:
    """Tests for trigger_build HTTP function."""

    def test_trigger_build_success(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"execution_id": "exec-123"}
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.post.return_value = mock_response

        result = cli.trigger_build(
            client=mock_client,
            goal="Add JWT auth",
            repo_path="/path/to/repo",
            repo_url="",
            config={"runtime": "claude_code"},
            additional_context="",
            server="http://localhost:8080",
        )

        assert result == "exec-123"
        call_args = mock_client.post.call_args
        assert (
            call_args[0][0]
            == "http://localhost:8080/api/v1/execute/async/swe-planner.build"
        )
        payload = call_args[1]["json"]
        assert payload["input"]["goal"] == "Add JWT auth"
        assert payload["input"]["repo_path"] == "/path/to/repo"
        assert payload["input"]["config"]["runtime"] == "claude_code"

    def test_trigger_build_with_repo_url(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"execution_id": "exec-456"}
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.post.return_value = mock_response

        result = cli.trigger_build(
            client=mock_client,
            goal="Fix bug",
            repo_path="",
            repo_url="https://github.com/user/repo.git",
            config={"runtime": "open_code"},
            additional_context="Extra info",
            server="http://localhost:8080",
        )

        assert result == "exec-456"
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["input"]["repo_url"] == "https://github.com/user/repo.git"
        assert payload["input"]["additional_context"] == "Extra info"

    def test_trigger_build_no_execution_id_raises(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"error": "Something went wrong"}
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.post.return_value = mock_response

        with pytest.raises(click.ClickException, match="No execution_id"):
            cli.trigger_build(
                client=mock_client,
                goal="Test",
                repo_path="/path",
                repo_url="",
                config={},
                additional_context="",
                server="http://localhost:8080",
            )


class TestTriggerPlan:
    """Tests for trigger_plan HTTP function."""

    def test_trigger_plan_success(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"execution_id": "plan-123"}
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.post.return_value = mock_response

        result = cli.trigger_plan(
            client=mock_client,
            goal="Add auth system",
            repo_path="/path/to/repo",
            additional_context="Use OAuth2",
            server="http://localhost:8080",
            config={"runtime": "claude_code", "models": {}},
        )

        assert result == "plan-123"
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["input"]["goal"] == "Add auth system"
        assert payload["input"]["ai_provider"] == "claude"

    def test_trigger_plan_open_code_runtime(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"execution_id": "plan-456"}
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.post.return_value = mock_response

        result = cli.trigger_plan(
            client=mock_client,
            goal="Test",
            repo_path="/path",
            additional_context="",
            server="http://localhost:8080",
            config={"runtime": "open_code", "models": {}},
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["input"]["ai_provider"] == "open_code"

    def test_trigger_plan_with_model_overrides(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"execution_id": "plan-789"}
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.post.return_value = mock_response

        cli.trigger_plan(
            client=mock_client,
            goal="Test",
            repo_path="/path",
            additional_context="",
            server="http://localhost:8080",
            config={
                "runtime": "claude_code",
                "models": {"pm": "opus", "architect": "sonnet"},
            },
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["input"]["pm_model"] == "opus"
        assert payload["input"]["architect_model"] == "sonnet"

    def test_trigger_plan_default_models(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"execution_id": "plan-000"}
        mock_response.raise_for_status = mock.Mock()

        mock_client = mock.Mock()
        mock_client.post.return_value = mock_response

        cli.trigger_plan(
            client=mock_client,
            goal="Test",
            repo_path="/path",
            additional_context="",
            server="http://localhost:8080",
            config={"runtime": "claude_code", "models": {}},
        )

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["input"]["pm_model"] == "sonnet"
        assert payload["input"]["architect_model"] == "sonnet"


class TestCliCommands:
    """Tests for CLI commands using CliRunner."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_cli_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli.cli, ["-h"])
        assert result.exit_code == 0
        assert "SWE-AF CLI client" in result.output
        assert "build" in result.output
        assert "plan" in result.output
        assert "status" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli.cli, ["--version"])
        assert result.exit_code == 0
        assert "swe, version" in result.output

    def test_cli_server_option(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "test-123"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.get.return_value = mock_response
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            runner.invoke(cli.cli, ["-s", "http://custom:9000", "list"])

            mock_client.get.assert_called()
            call_url = mock_client.get.call_args[0][0]
            assert "http://custom:9000" in call_url

    def test_build_command_defaults_to_cwd(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "build-123"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["build", "Add feature"])

            assert result.exit_code == 0
            assert "Build started: build-123" in result.output

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["input"]["repo_path"] == os.getcwd()

    def test_build_command_with_path(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "build-456"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(
                cli.cli, ["build", "Add feature", "--path", "/custom/path"]
            )

            assert result.exit_code == 0
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["input"]["repo_path"] == "/custom/path"

    def test_build_command_with_models(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "build-789"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(
                cli.cli,
                [
                    "build",
                    "Add feature",
                    "--model",
                    "coder=opus",
                    "--model",
                    "qa=sonnet",
                ],
            )

            assert result.exit_code == 0
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["input"]["config"]["models"]["coder"] == "opus"
            assert payload["input"]["config"]["models"]["qa"] == "sonnet"

    def test_build_command_with_learning(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "build-learn"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["build", "Add feature", "--learning"])

            assert result.exit_code == 0
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["input"]["config"]["enable_learning"] is True

    def test_build_command_verbose_output(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "build-verbose"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["build", "Add feature", "--verbose"])

            assert result.exit_code == 0
            assert "Server:" in result.output
            assert "Goal:" in result.output
            assert "Config:" in result.output

    def test_build_command_connect_error(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["build", "Add feature"])

            assert result.exit_code != 0
            assert "Cannot connect to AgentField" in result.output

    def test_build_command_http_error(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Server error", request=mock.Mock(), response=mock_response
            )
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["build", "Add feature"])

            assert result.exit_code != 0
            assert "HTTP 500" in result.output

    def test_plan_command_success(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "plan-123"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["plan", "Add auth system"])

            assert result.exit_code == 0
            assert "Plan started: plan-123" in result.output

    def test_status_command_success(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {
                "execution_id": "abc123",
                "status": "completed",
                "node_id": "swe-planner",
                "reasoner": "build",
            }
            mock_response.raise_for_status = mock.Mock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["status", "abc123"])

            assert result.exit_code == 0
            assert "Execution: abc123" in result.output
            assert "Status:    completed" in result.output

    def test_status_command_json_output(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {
                "execution_id": "abc123",
                "status": "completed",
            }
            mock_response.raise_for_status = mock.Mock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["status", "abc123", "--json-output"])

            assert result.exit_code == 0
            output_json = json.loads(result.output.strip())
            assert output_json["execution_id"] == "abc123"

    def test_list_command_success(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {
                "executions": [
                    {
                        "execution_id": "abc123def456",
                        "status": "completed",
                        "reasoner": "build",
                        "created_at": "2024-01-01T00:00:00",
                    },
                    {
                        "execution_id": "xyz789",
                        "status": "running",
                        "reasoner": "plan",
                        "created_at": "2024-01-02T00:00:00",
                    },
                ]
            }
            mock_response.raise_for_status = mock.Mock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["list"])

            assert result.exit_code == 0
            assert "Recent executions" in result.output
            assert "abc123def456" in result.output

    def test_list_command_empty(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"executions": []}
            mock_response.raise_for_status = mock.Mock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["list"])

            assert result.exit_code == 0
            assert "No executions found" in result.output

    def test_agents_command_success(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {
                "nodes": [
                    {
                        "node_id": "swe-planner",
                        "version": "0.1.0",
                        "description": "SWE planning agent",
                        "reasoners": ["build", "plan", "run_coder"],
                    }
                ]
            }
            mock_response.raise_for_status = mock.Mock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["agents"])

            assert result.exit_code == 0
            assert "swe-planner" in result.output
            assert "build, plan, run_coder" in result.output

    def test_agents_command_empty(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"nodes": []}
            mock_response.raise_for_status = mock.Mock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["agents"])

            assert result.exit_code == 0
            assert "No agents registered" in result.output

    def test_agents_command_404(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.status_code = 404
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Not found", request=mock.Mock(), response=mock_response
            )
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["agents"])

            assert result.exit_code == 0
            assert "not available on this server version" in result.output

    def test_cancel_command_success(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"status": "cancelled"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["cancel", "abc123"])

            assert result.exit_code == 0
            assert "abc123 cancelled" in result.output

    def test_cancel_command_not_found(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.status_code = 404
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Not found", request=mock.Mock(), response=mock_response
            )
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(cli.cli, ["cancel", "nonexistent"])

            assert result.exit_code != 0
            assert "not found" in result.output

    def test_call_command_with_json(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "call-123"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(
                cli.cli,
                [
                    "call",
                    "run_coder",
                    '{"issue": {"title": "Fix bug"}, "repo_path": "/path"}',
                ],
            )

            assert result.exit_code == 0
            assert "Called swe-planner.run_coder" in result.output
            assert "call-123" in result.output

    def test_call_command_with_params(self, runner: CliRunner) -> None:
        with mock.patch.object(cli, "get_client") as mock_get_client:
            mock_client = mock.Mock()
            mock_response = mock.Mock()
            mock_response.json.return_value = {"execution_id": "call-456"}
            mock_response.raise_for_status = mock.Mock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value.__enter__ = mock.Mock(return_value=mock_client)
            mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

            result = runner.invoke(
                cli.cli,
                [
                    "call",
                    "run_coder",
                    "--param",
                    "repo_path=/my/path",
                    "--param",
                    "timeout=30",
                ],
            )

            assert result.exit_code == 0
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["input"]["repo_path"] == "/my/path"
            assert payload["input"]["timeout"] == 30

    def test_call_command_invalid_json(self, runner: CliRunner) -> None:
        result = runner.invoke(cli.cli, ["call", "run_coder", "not valid json"])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_call_command_invalid_param(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli.cli, ["call", "run_coder", "--param", "no_equals_sign"]
        )

        assert result.exit_code != 0
        assert "must be key=value" in result.output


class TestLogsCommand:
    """Tests for logs CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_logs_no_artifacts_dir(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli.cli, ["logs", tmpdir])

            assert result.exit_code != 0
            assert "No execution logs found" in result.output

    def test_logs_with_checkpoint(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            execution_dir = Path(tmpdir) / ".artifacts" / "execution"
            execution_dir.mkdir(parents=True)

            checkpoint = {"build_id": "test-123", "status": "completed"}
            checkpoint_path = execution_dir / "checkpoint.json"
            checkpoint_path.write_text(json.dumps(checkpoint))

            result = runner.invoke(cli.cli, ["logs", tmpdir])

            assert result.exit_code == 0
            assert "Build Checkpoint" in result.output
            assert "test-123" in result.output

    def test_logs_with_issue_logs(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            execution_dir = Path(tmpdir) / ".artifacts" / "execution"
            execution_dir.mkdir(parents=True)

            issue_log = {"issue_name": "Add auth", "status": "completed"}
            issue_path = execution_dir / "issue-001.json"
            issue_path.write_text(json.dumps(issue_log))

            result = runner.invoke(cli.cli, ["logs", tmpdir])

            assert result.exit_code == 0
            assert "Issue Logs" in result.output
            assert "Add auth" in result.output

    def test_logs_empty_execution_dir(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            execution_dir = Path(tmpdir) / ".artifacts" / "execution"
            execution_dir.mkdir(parents=True)

            result = runner.invoke(cli.cli, ["logs", tmpdir])

            assert result.exit_code == 0
            assert "No logs found" in result.output


class TestResumeCommand:
    """Tests for resume CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_resume_success(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(cli, "get_client") as mock_get_client:
                mock_client = mock.Mock()
                mock_response = mock.Mock()
                mock_response.json.return_value = {"execution_id": "resume-123"}
                mock_response.raise_for_status = mock.Mock()
                mock_client.post.return_value = mock_response
                mock_get_client.return_value.__enter__ = mock.Mock(
                    return_value=mock_client
                )
                mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

                result = runner.invoke(cli.cli, ["resume", tmpdir])

                assert result.exit_code == 0
                assert "Resume started: resume-123" in result.output


class TestExecuteCommand:
    """Tests for execute CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_execute_no_plan(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli.cli, ["execute", tmpdir])

            assert result.exit_code != 0
            assert "No plan found" in result.output

    def test_execute_with_plan(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_dir = Path(tmpdir) / ".artifacts" / "plan"
            plan_dir.mkdir(parents=True)
            (plan_dir / "prd.md").write_text("# PRD")

            with mock.patch.object(cli, "get_client") as mock_get_client:
                mock_client = mock.Mock()
                mock_response = mock.Mock()
                mock_response.json.return_value = {"execution_id": "exec-123"}
                mock_response.raise_for_status = mock.Mock()
                mock_client.post.return_value = mock_response
                mock_get_client.return_value.__enter__ = mock.Mock(
                    return_value=mock_client
                )
                mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

                result = runner.invoke(cli.cli, ["execute", tmpdir])

                assert result.exit_code == 0
                assert "Execution started: exec-123" in result.output


class TestVerifyCommand:
    """Tests for verify CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_verify_no_prd(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli.cli, ["verify", tmpdir])

            assert result.exit_code != 0
            assert "No PRD found" in result.output

    def test_verify_with_prd(self, runner: CliRunner) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_dir = Path(tmpdir) / ".artifacts" / "plan"
            plan_dir.mkdir(parents=True)
            (plan_dir / "prd.md").write_text("# PRD")

            with mock.patch.object(cli, "get_client") as mock_get_client:
                mock_client = mock.Mock()
                mock_response = mock.Mock()
                mock_response.json.return_value = {"execution_id": "verify-123"}
                mock_response.raise_for_status = mock.Mock()
                mock_client.post.return_value = mock_response
                mock_get_client.return_value.__enter__ = mock.Mock(
                    return_value=mock_client
                )
                mock_get_client.return_value.__exit__ = mock.Mock(return_value=False)

                result = runner.invoke(cli.cli, ["verify", tmpdir])

                assert result.exit_code == 0
                assert "Verification started: verify-123" in result.output


class TestPollExecution:
    """Tests for poll_execution function."""

    def test_poll_disabled_returns_immediately(self) -> None:
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.json.return_value = {"execution_id": "test", "status": "running"}
        mock_response.raise_for_status = mock.Mock()
        mock_client.get.return_value = mock_response

        result = cli.poll_execution(
            mock_client, "test", "http://localhost:8080", poll=False
        )

        assert result["status"] == "running"
        mock_client.get.assert_called_once()

    def test_poll_until_completed(self) -> None:
        mock_client = mock.Mock()
        responses = [
            {"execution_id": "test", "status": "running"},
            {"execution_id": "test", "status": "running"},
            {"execution_id": "test", "status": "completed"},
        ]

        def get_side_effect(*args, **kwargs):
            mock_response = mock.Mock()
            mock_response.json.return_value = responses.pop(0)
            mock_response.raise_for_status = mock.Mock()
            return mock_response

        mock_client.get.side_effect = get_side_effect

        with mock.patch("swe_af.cli.POLL_INTERVAL", 0.01):
            with mock.patch("click.echo"):
                result = cli.poll_execution(
                    mock_client, "test", "http://localhost:8080", poll=True
                )

        assert result["status"] == "completed"
        assert mock_client.get.call_count == 3

    def test_poll_until_failed(self) -> None:
        mock_client = mock.Mock()
        responses = [
            {"execution_id": "test", "status": "running"},
            {"execution_id": "test", "status": "failed"},
        ]

        def get_side_effect(*args, **kwargs):
            mock_response = mock.Mock()
            mock_response.json.return_value = responses.pop(0)
            mock_response.raise_for_status = mock.Mock()
            return mock_response

        mock_client.get.side_effect = get_side_effect

        with mock.patch("swe_af.cli.POLL_INTERVAL", 0.01):
            with mock.patch("click.echo"):
                result = cli.poll_execution(
                    mock_client, "test", "http://localhost:8080", poll=True
                )

        assert result["status"] == "failed"


class TestDefaultServer:
    """Tests for DEFAULT_SERVER configuration."""

    def test_default_server_without_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            from importlib import reload

            reload(cli)
            assert cli.DEFAULT_SERVER == "http://localhost:8080"

    def test_default_server_with_env(self) -> None:
        with mock.patch.dict(
            os.environ, {"AGENTFIELD_SERVER": "http://custom:9000"}, clear=True
        ):
            from importlib import reload

            reload(cli)
            assert cli.DEFAULT_SERVER == "http://custom:9000"
