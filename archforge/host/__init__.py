"""Host multi-agent system integration (spec §3, §4).

ArchForge does not depend on any specific agentic framework. A host MAS
implements the `HostMAS` contract: it can build an executable pipeline from a
`Spec` and run a `Task` through it, emitting one `Step` per agent. The
`TracingMiddleware` attaches as both observer (records steps to TraceStore) and
config bridge (applies the live Spec's prompt/model/knobs to each agent at
invoke time — so evolving the pipeline is swapping which Spec the host uses).
"""

from __future__ import annotations

from archforge.host.base import Agent, AgentResponse, HostMAS, Runnable, Task
from archforge.host.fake import (
    FakeAgent, FakeHostMAS, FakeRuleAgent, FakeRetrieverAgent, FakeToolAgent,
)

__all__ = ["Agent", "AgentResponse", "HostMAS", "Runnable", "Task",
           "FakeHostMAS", "FakeAgent", "FakeRuleAgent", "FakeRetrieverAgent",
           "FakeToolAgent"]
