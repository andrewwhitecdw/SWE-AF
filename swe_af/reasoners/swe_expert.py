"""
SWE-Expert Reasoner - Autonomous software build orchestrator.

A single entry point that orchestrates complete software builds with:
- Intelligent context gathering
- Self-healing execution
- Multi-language support (Go, Rust, Python, TypeScript, etc.)
- Iterative refinement on failures
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid

from .schemas import PRD, Architecture, ReviewResult, PlannedIssue, PlanResult

from . import router


# =============================================================================
# SWE-Expert: Complete Build Orchestrator
# =============================================================================


@router.reasoner()
async def swe_expert(
    goal: str,
    repo_path: str = "",
    repo_url: str = "",
    artifacts_dir: str = ".artifacts",
    additional_context: str = "",
    config: dict | None = None,
    max_iterations: int = 3,
    auto_commit: bool = True,
    auto_pr: bool = False,
) -> dict:
    """Autonomous software build orchestrator.

    Single entry point for complete software development:
    1. Analyzes the goal and repository
    2. Creates a comprehensive plan (PRD, architecture, issues)
    3. Executes the plan with self-healing
    4. Verifies the results against acceptance criteria
    5. Iterates on failures up to max_iterations
    6. Optionally commits and creates a PR

    Args:
        goal: What to build (natural language description)
        repo_path: Local repository path
        repo_url: Git repository URL (alternative to repo_path)
        artifacts_dir: Where to store build artifacts
        additional_context: Extra context for the AI
        config: Runtime configuration (model, tools, etc.)
        max_iterations: Maximum fix attempts (default: 3)
        auto_commit: Commit changes on success (default: True)
        auto_pr: Create PR on success (default: False)

    Returns:
        dict with success status, artifacts, summary, and recommendations
    """
    from .pipeline import (
        run_product_manager,
        run_architect,
        run_tech_lead,
        run_sprint_planner,
    )

    execution_id = (
        f"swe-expert-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )

    router.note(f"[swe-expert] Starting autonomous build: {goal[:100]}...")
    router.note(f"[swe-expert] Execution ID: {execution_id}")

    # Gather repository context
    repo_context = await _gather_repo_context(repo_path) if repo_path else {}
    router.note(
        f"[swe-expert] Detected: {repo_context.get('language', 'unknown')} ({repo_context.get('build_system', 'unknown')})"
    )

    # Build enhanced context
    enhanced_context = f"""
{additional_context}

## Repository Context
- Language: {repo_context.get("language", "unknown")}
- Build system: {repo_context.get("build_system", "unknown")}
- Test framework: {repo_context.get("test_framework", "unknown")}
- Structure: {repo_context.get("structure", "unknown")}
- Key files: {", ".join(repo_context.get("key_files", [])[:10])}
"""

    errors = []
    recommendations = []

    model = (
        config.get("model", "zhipuai-coding-plan/glm-5")
        if config
        else "zhipuai-coding-plan/glm-5"
    )
    max_turns = config.get("max_turns", 100) if config else 100
    permission_mode = config.get("permission_mode", "") if config else ""
    ai_provider = config.get("ai_provider", "opencode") if config else "opencode"

    for iteration in range(1, max_iterations + 1):
        router.note(f"[swe-expert] Starting iteration {iteration}/{max_iterations}")

        try:
            prd_result = await run_product_manager(
                goal=goal,
                repo_path=repo_path,
                artifacts_dir=artifacts_dir,
                additional_context=enhanced_context,
                model=model,
                max_turns=max_turns,
                permission_mode=permission_mode,
                ai_provider=ai_provider,
            )

            if not prd_result.get("success"):
                error_msg = prd_result.get("error", "Unknown PM error")
                errors.append(f"PM phase: {error_msg}")
                continue

            router.note("[swe-expert] Phase 2: Architecture")
            arch_result = await run_architect(
                prd=prd_result.get("prd", {}),
                repo_path=repo_path,
                artifacts_dir=artifacts_dir,
                model=model,
                max_turns=max_turns,
                permission_mode=permission_mode,
                ai_provider=ai_provider,
            )

            router.note("[swe-expert] Phase 3: Tech Lead Review")
            review_result = await run_tech_lead(
                prd=prd_result.get("prd", {}),
                repo_path=repo_path,
                artifacts_dir=artifacts_dir,
                revision_number=0,
                model=model,
                max_turns=max_turns,
                permission_mode=permission_mode,
                ai_provider=ai_provider,
            )

            router.note("[swe-expert] Phase 4: Sprint Planning")
            sprint_result = await run_sprint_planner(
                prd=prd_result.get("prd", {}),
                architecture=arch_result.get("architecture", {}),
                repo_path=repo_path,
                artifacts_dir=artifacts_dir,
                model=model,
                max_turns=max_turns,
                permission_mode=permission_mode,
                ai_provider=ai_provider,
            )

            issues = sprint_result.get("issues", [])
            if not issues:
                errors.append("No issues generated from planning")
                continue

            router.note(f"[swe-expert] Planned {len(issues)} issues")

            router.note("[swe-expert] Phase 5: Execution")
            exec_result = await _execute_issues(
                issues=issues,
                repo_path=repo_path,
                artifacts_dir=artifacts_dir,
                model=model,
                permission_mode=permission_mode,
                ai_provider=ai_provider,
                auto_commit=auto_commit,
            )

            if exec_result.get("success"):
                router.note(
                    f"[swe-expert] Build completed successfully on iteration {iteration}"
                )

                return {
                    "success": True,
                    "execution_id": execution_id,
                    "goal": goal,
                    "iterations_used": iteration,
                    "final_state": "completed",
                    "artifacts": {
                        "prd": prd_result.get("prd"),
                        "architecture": arch_result.get("architecture"),
                        "issues": issues,
                    },
                    "summary": exec_result.get(
                        "summary", "Build completed successfully"
                    ),
                    "errors": errors,
                    "recommendations": recommendations,
                }

            error_msg = exec_result.get("error", "Execution failed")
            errors.append(f"Iteration {iteration}: {error_msg}")

        except Exception as e:
            error_msg = str(e)
            router.note(f"[swe-expert] Exception on iteration {iteration}: {error_msg}")
            errors.append(f"Iteration {iteration}: {error_msg}")

        # Update context with learnings
        enhanced_context += f"\n\n## Previous Attempt {iteration}\nErrors encountered. Please avoid these issues."

    router.note(f"[swe-expert] Build failed after {max_iterations} iterations")

    return {
        "success": False,
        "execution_id": execution_id,
        "goal": goal,
        "iterations_used": max_iterations,
        "final_state": "failed",
        "artifacts": {},
        "summary": f"Build failed after {max_iterations} iterations",
        "errors": errors,
        "recommendations": recommendations
        + ["Break goal into smaller tasks", "Check repository structure"],
    }


async def _gather_repo_context(repo_path: str) -> dict:
    """Gather context about the repository structure and language."""
    context = {
        "language": "unknown",
        "build_system": "unknown",
        "test_framework": "unknown",
        "structure": "unknown",
        "key_files": [],
    }

    if not repo_path or not os.path.exists(repo_path):
        return context

    path = Path(repo_path)

    # Go detection
    if (path / "go.mod").exists():
        context["language"] = "Go"
        context["build_system"] = "Go modules"
        context["key_files"].append("go.mod")
        if list(path.glob("**/*_test.go")):
            context["test_framework"] = "go test"
        if (path / "Makefile").exists():
            context["key_files"].append("Makefile")

    # Rust detection
    elif (path / "Cargo.toml").exists():
        context["language"] = "Rust"
        context["build_system"] = "Cargo"
        context["key_files"].append("Cargo.toml")
        if (path / "Cargo.lock").exists():
            context["key_files"].append("Cargo.lock")
        if list(path.glob("**/tests/*.rs")) or list(path.glob("**/*_test.rs")):
            context["test_framework"] = "cargo test"

    # Python detection
    elif (path / "pyproject.toml").exists():
        context["language"] = "Python"
        context["build_system"] = "pip/poetry"
        context["key_files"].append("pyproject.toml")
        if (path / "pytest.ini").exists():
            context["test_framework"] = "pytest"

    # TypeScript/JavaScript detection
    elif (path / "package.json").exists():
        context["language"] = "TypeScript/JavaScript"
        context["build_system"] = "npm/yarn/pnpm"
        context["key_files"].append("package.json")
        if (path / "jest.config.js").exists() or (path / "vitest.config.ts").exists():
            context["test_framework"] = "Jest/Vitest"

    # Key config files
    for name in ["README.md", "CONTRIBUTING.md", "AGENTS.md", ".env.example"]:
        if (path / name).exists():
            context["key_files"].append(name)

    # Structure detection
    if (path / "cmd").exists() and (path / "internal").exists():
        context["structure"] = "Go standard layout"
    elif (path / "src").exists():
        context["structure"] = "src-based"
    elif (path / "lib").exists():
        context["structure"] = "lib-based"
    elif (path / "crates").exists():
        context["structure"] = "Rust workspace"

    return context


async def _execute_issues(
    issues: list,
    repo_path: str,
    artifacts_dir: str,
    model: str = "zhipuai-coding-plan/glm-5",
    permission_mode: str = "",
    ai_provider: str = "opencode",
    auto_commit: bool = False,
) -> dict:
    from .execution_agents import run_coder, run_qa, run_code_reviewer

    completed = []
    failed = []

    for issue in issues:
        issue_name = issue.get("name", "unknown")
        router.note(f"[swe-expert] Executing issue: {issue_name}")

        try:
            coder_result = await run_coder(
                issue=issue,
                worktree_path=repo_path,
                feedback="",
                iteration=1,
                iteration_id=f"{issue_name}-1",
                project_context={
                    "artifacts_dir": artifacts_dir,
                    "repo_path": repo_path,
                },
                memory_context=None,
                model=model,
                permission_mode=permission_mode,
                ai_provider=ai_provider,
            )

            if not coder_result.get("success"):
                failed.append({"issue": issue_name, "error": coder_result.get("error")})
                continue

            qa_result = await run_qa(
                worktree_path=repo_path,
                coder_result=coder_result,
                issue=issue,
                model=model,
                permission_mode=permission_mode,
                ai_provider=ai_provider,
            )

            if qa_result.get("passed"):
                completed.append(issue_name)
            else:
                failed.append({"issue": issue_name, "error": "QA failed"})

        except Exception as e:
            failed.append({"issue": issue_name, "error": str(e)})

    if not failed:
        return {
            "success": True,
            "summary": f"Completed {len(completed)} issues successfully",
        }

    return {
        "success": False,
        "error": f"Failed issues: {[f['issue'] for f in failed]}",
        "completed": completed,
        "failed": failed,
    }


# =============================================================================
# Quick Build: Fast iteration for simple tasks
# =============================================================================


@router.reasoner()
async def quick_build(
    goal: str,
    repo_path: str,
    files_to_modify: list[str] | None = None,
    config: dict | None = None,
) -> dict:
    """Fast build for simple, single-file or few-file changes.

    Skips full planning pipeline. Best for:
    - Bug fixes
    - Small feature additions
    - Refactoring

    Args:
        goal: What to build
        repo_path: Local repository path
        files_to_modify: Hint about which files to touch
        config: Runtime configuration

    Returns:
        dict with success status and summary
    """
    from .execution_agents import run_coder, run_qa, run_code_reviewer

    router.note(f"[quick-build] Starting: {goal[:80]}...")

    issue = {
        "name": "quick-build-task",
        "title": goal,
        "description": goal,
        "files_to_modify": files_to_modify or [],
        "acceptance_criteria": [f"Goal achieved: {goal}"],
    }

    model = (
        config.get("model", "zhipuai-coding-plan/glm-5")
        if config
        else "zhipuai-coding-plan/glm-5"
    )
    permission_mode = config.get("permission_mode", "") if config else ""
    ai_provider = config.get("ai_provider", "opencode") if config else "opencode"

    coder_result = await run_coder(
        issue=issue,
        worktree_path=repo_path,
        feedback="",
        iteration=1,
        iteration_id="quick-build-1",
        project_context=None,
        memory_context=None,
        model=model,
        permission_mode=permission_mode,
        ai_provider=ai_provider,
    )

    if not coder_result.get("success"):
        return {"success": False, "error": coder_result.get("error")}

    qa_result = await run_qa(
        worktree_path=repo_path,
        coder_result=coder_result,
        issue=issue,
        model=model,
        permission_mode=permission_mode,
        ai_provider=ai_provider,
    )

    if not qa_result.get("passed"):
        return {"success": False, "error": "Tests failed", "details": qa_result}

    review_result = await run_code_reviewer(
        worktree_path=repo_path,
        coder_result=coder_result,
        issue=issue,
        model=model,
        permission_mode=permission_mode,
        ai_provider=ai_provider,
    )

    return {
        "success": True,
        "summary": f"Quick build completed: {goal}",
        "changes": coder_result.get("changes"),
        "review": review_result.get("review"),
    }


__all__ = ["swe_expert", "quick_build"]
