from agentfield import AgentRouter

router = AgentRouter(tags=["swe-planner"])

from . import execution_agents  # noqa: E402, F401 — registers execution reasoners
from . import pipeline  # noqa: E402, F401 — registers planning reasoners
from . import dagger_runner  # noqa: E402, F401 — registers Dagger CI/CD reasoners
from . import swe_expert  # noqa: E402, F401 — registers SWE-Expert orchestrator
from . import rust  # noqa: E402, F401 — registers Rust language reasoners

__all__ = ["router"]
