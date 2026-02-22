"""
Rust Language Reasoners - Specialized reasoners for Rust projects.

Provides Rust-specific build, test, and analysis capabilities:
- Cargo command execution
- Rust analyzer integration
- Clippy linting
- Rustdoc generation
- Fmt checking
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional
import json

from . import router


# =============================================================================
# Cargo Reasoners
# =============================================================================


@router.reasoner()
async def cargo_check(
    repo_path: str,
    all_features: bool = True,
    all_targets: bool = True,
) -> dict:
    """Run cargo check to verify compilation without producing binaries.

    Args:
        repo_path: Path to Rust project
        all_features: Check with all features enabled
        all_targets: Check all targets (including tests/benches)

    Returns:
        dict with success status and any errors/warnings
    """
    cmd = ["cargo", "check"]
    if all_features:
        cmd.append("--all-features")
    if all_targets:
        cmd.append("--all-targets")

    router.note(f"[rust] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "cargo check timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.reasoner()
async def cargo_build(
    repo_path: str,
    release: bool = False,
    all_features: bool = True,
) -> dict:
    """Build the Rust project.

    Args:
        repo_path: Path to Rust project
        release: Build in release mode (optimized)
        all_features: Build with all features

    Returns:
        dict with success status and build artifacts
    """
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    if all_features:
        cmd.append("--all-features")

    router.note(f"[rust] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min for builds
        )

        # Find built artifacts
        target_dir = Path(repo_path) / "target"
        artifacts = []
        if release:
            bin_dir = target_dir / "release"
        else:
            bin_dir = target_dir / "debug"

        if bin_dir.exists():
            artifacts = [
                f.name
                for f in bin_dir.iterdir()
                if f.is_file() and os.access(f, os.X_OK)
            ]

        return {
            "success": result.returncode == 0,
            "artifacts": artifacts,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "cargo build timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.reasoner()
async def cargo_test(
    repo_path: str,
    test_filter: str = "",
    release: bool = False,
    no_fail_fast: bool = True,
) -> dict:
    """Run Rust tests.

    Args:
        repo_path: Path to Rust project
        test_filter: Optional filter for specific tests
        release: Run tests in release mode
        no_fail_fast: Continue running tests after failures

    Returns:
        dict with test results
    """
    cmd = ["cargo", "test"]
    if release:
        cmd.append("--release")
    if no_fail_fast:
        cmd.append("--no-fail-fast")
    if test_filter:
        cmd.append(test_filter)

    router.note(f"[rust] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Parse test summary
        output = result.stdout + result.stderr
        passed = 0
        failed = 0
        ignored = 0

        for line in output.split("\n"):
            if "test result:" in line:
                # Parse "test result: ok. X passed; Y failed; Z ignored"
                parts = line.split("test result:")[1] if "test result:" in line else ""
                for part in parts.split(";"):
                    part = part.strip()
                    if "passed" in part:
                        passed = int(part.split()[0] or 0)
                    elif "failed" in part:
                        failed = int(part.split()[0] or 0)
                    elif "ignored" in part:
                        ignored = int(part.split()[0] or 0)

        return {
            "success": result.returncode == 0,
            "passed": passed,
            "failed": failed,
            "ignored": ignored,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "cargo test timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.reasoner()
async def cargo_clippy(
    repo_path: str,
    all_features: bool = True,
    deny_warnings: bool = True,
    fix: bool = False,
) -> dict:
    """Run Clippy linter on Rust code.

    Args:
        repo_path: Path to Rust project
        all_features: Lint with all features
        deny_warnings: Treat warnings as errors
        fix: Auto-fix issues where possible

    Returns:
        dict with lint results and fix suggestions
    """
    cmd = ["cargo", "clippy"]
    if all_features:
        cmd.extend(["--all-features"])
    if deny_warnings:
        cmd.extend(["--", "-D", "warnings"])
    if fix:
        cmd.append("--fix")  # Note: requires --allow-dirty for non-interactive

    router.note(f"[rust] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Parse warnings and errors
        warnings = []
        errors = []
        for line in (result.stdout + result.stderr).split("\n"):
            if "warning:" in line.lower():
                warnings.append(line)
            elif "error:" in line.lower():
                errors.append(line)

        return {
            "success": result.returncode == 0,
            "warnings": warnings,
            "errors": errors,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "cargo clippy timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.reasoner()
async def cargo_fmt(
    repo_path: str,
    check_only: bool = True,
) -> dict:
    """Run rustfmt on Rust code.

    Args:
        repo_path: Path to Rust project
        check_only: Only check formatting, don't modify files

    Returns:
        dict with formatting status and any issues
    """
    cmd = ["cargo", "fmt"]
    if check_only:
        cmd.append("--check")

    router.note(f"[rust] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Parse diff output if check_only
        unformatted = []
        if result.stdout:
            for line in result.stdout.split("\n"):
                if line.startswith("Diff in"):
                    unformatted.append(line)

        return {
            "success": result.returncode == 0,
            "formatted": result.returncode == 0,
            "unformatted_files": unformatted,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "cargo fmt timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Rust Analyzer Reasoners
# =============================================================================


@router.reasoner()
async def rust_analyzer_diagnostics(
    repo_path: str,
) -> dict:
    """Get diagnostics from rust-analyzer LSP.

    Returns type errors, warnings, hints from rust-analyzer.
    Requires rust-analyzer to be installed.

    Args:
        repo_path: Path to Rust project

    Returns:
        dict with diagnostics by file
    """
    # Check if rust-analyzer is available
    try:
        result = subprocess.run(
            ["rust-analyzer", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"success": False, "error": "rust-analyzer not found"}
    except Exception as e:
        return {"success": False, "error": f"rust-analyzer not available: {e}"}

    # Note: Full rust-analyzer integration requires LSP client setup
    # This is a simplified check using cargo check with JSON output
    result = subprocess.run(
        ["cargo", "check", "--message-format=json"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=300,
    )

    diagnostics = {"errors": [], "warnings": []}

    for line in result.stdout.split("\n"):
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
            if msg.get("reason") == "compiler-message":
                level = msg.get("message", {}).get("level", "")
                text = msg.get("message", {}).get("message", "")
                if level == "error":
                    diagnostics["errors"].append(text)
                elif level == "warning":
                    diagnostics["warnings"].append(text)
        except json.JSONDecodeError:
            continue

    return {
        "success": len(diagnostics["errors"]) == 0,
        "diagnostics": diagnostics,
    }


@router.reasoner()
async def rustdoc_build(
    repo_path: str,
    open_result: bool = False,
) -> dict:
    """Generate Rust documentation.

    Args:
        repo_path: Path to Rust project
        open_result: Open docs in browser after generation

    Returns:
        dict with doc location
    """
    cmd = ["cargo", "doc", "--no-deps"]

    router.note(f"[rust] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=600,
        )

        doc_path = Path(repo_path) / "target" / "doc" / "index.html"

        return {
            "success": result.returncode == 0,
            "doc_path": str(doc_path) if doc_path.exists() else None,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "cargo doc timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Workspace Reasoners
# =============================================================================


@router.reasoner()
async def cargo_workspace_info(
    repo_path: str,
) -> dict:
    """Get information about a Cargo workspace.

    Args:
        repo_path: Path to Cargo workspace

    Returns:
        dict with workspace structure info
    """
    cargo_toml = Path(repo_path) / "Cargo.toml"

    if not cargo_toml.exists():
        return {"success": False, "error": "No Cargo.toml found"}

    # Try to get workspace members
    try:
        result = subprocess.run(
            ["cargo", "metadata", "--format-version=1", "--no-deps"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            workspace_members = metadata.get("workspace_members", [])
            packages = metadata.get("packages", [])

            return {
                "success": True,
                "is_workspace": len(workspace_members) > 1,
                "members": [p.split()[0] for p in workspace_members],
                "package_count": len(packages),
            }
    except Exception as e:
        pass

    # Fallback: check if it's a single crate
    return {
        "success": True,
        "is_workspace": False,
        "members": [Path(repo_path).name],
        "package_count": 1,
    }


@router.reasoner()
async def rust_audit(
    repo_path: str,
) -> dict:
    """Run cargo-audit to check for security vulnerabilities.

    Requires cargo-audit to be installed.

    Args:
        repo_path: Path to Rust project

    Returns:
        dict with vulnerability report
    """
    try:
        result = subprocess.run(
            ["cargo", "audit"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,
        )

        vulnerabilities = []
        for line in (result.stdout + result.stderr).split("\n"):
            if "Crate:" in line or "Version:" in line or "ID:" in line:
                vulnerabilities.append(line)

        return {
            "success": result.returncode == 0,
            "has_vulnerabilities": result.returncode != 0,
            "vulnerabilities": vulnerabilities,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "cargo-audit not installed. Run: cargo install cargo-audit",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Combined Rust CI Reasoner
# =============================================================================


@router.reasoner()
async def rust_ci(
    repo_path: str,
    run_tests: bool = True,
    run_clippy: bool = True,
    check_format: bool = True,
    run_audit: bool = False,
) -> dict:
    """Run complete Rust CI pipeline.

    Runs check, build, test, clippy, and fmt checks.

    Args:
        repo_path: Path to Rust project
        run_tests: Include cargo test
        run_clippy: Include clippy linting
        check_format: Include fmt check
        run_audit: Include security audit

    Returns:
        dict with combined CI results
    """
    router.note(f"[rust-ci] Starting CI pipeline for {repo_path}")

    results = {}
    all_passed = True

    # Check
    check_result = await cargo_check(repo_path)
    results["check"] = {"passed": check_result["success"]}
    if not check_result["success"]:
        results["check"]["errors"] = check_result.get("stderr", "")
        all_passed = False

    # Build
    build_result = await cargo_build(repo_path, release=False)
    results["build"] = {"passed": build_result["success"]}
    if not build_result["success"]:
        results["build"]["errors"] = build_result.get("stderr", "")
        all_passed = False

    # Test
    if run_tests:
        test_result = await cargo_test(repo_path)
        results["test"] = {
            "passed": test_result["success"],
            "passed_count": test_result.get("passed", 0),
            "failed_count": test_result.get("failed", 0),
        }
        if not test_result["success"]:
            all_passed = False

    # Clippy
    if run_clippy:
        clippy_result = await cargo_clippy(repo_path)
        results["clippy"] = {
            "passed": clippy_result["success"],
            "warnings": len(clippy_result.get("warnings", [])),
            "errors": len(clippy_result.get("errors", [])),
        }
        if not clippy_result["success"]:
            all_passed = False

    # Fmt
    if check_format:
        fmt_result = await cargo_fmt(repo_path, check_only=True)
        results["fmt"] = {"passed": fmt_result["success"]}
        if not fmt_result["success"]:
            all_passed = False

    # Audit
    if run_audit:
        audit_result = await rust_audit(repo_path)
        results["audit"] = {
            "passed": audit_result["success"],
            "vulnerabilities": len(audit_result.get("vulnerabilities", [])),
        }
        if not audit_result["success"]:
            all_passed = False

    return {
        "success": all_passed,
        "summary": "All CI checks passed" if all_passed else "Some CI checks failed",
        "results": results,
    }


__all__ = [
    "cargo_check",
    "cargo_build",
    "cargo_test",
    "cargo_clippy",
    "cargo_fmt",
    "rust_analyzer_diagnostics",
    "rustdoc_build",
    "cargo_workspace_info",
    "rust_audit",
    "rust_ci",
]
