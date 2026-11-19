"""A test double for the chat model. It replays a fixed list of tool calls.

This is not an agent run and no result of it may be reported as agent behavior. It only exercises the
graph structure, the counters and the checks without any LLM.
"""

from langchain_core.messages import AIMessage


class ScriptedLLM:
    """Replays planned tool calls in order, and records what it was given."""

    def __init__(self, script):
        """Store the planned calls, each a tuple (tool name, arguments) or an Exception to raise."""
        self.script = list(script)
        self.seen_contexts = []
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        """Mimic bind_tools of a chat model and return this object."""
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        """Return the next planned call as an AIMessage and keep the messages for assertions."""
        self.seen_contexts.append(messages)
        if not self.script:
            raise AssertionError("the graph asked the LLM more often than planned")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        name, args = step
        if name != "submit_decision":
            # every tool except submit_decision has a reason, like a real call of the LLM
            args = dict(args)
            args.setdefault("reason", "scripted test step")
        call_id = f"call_{len(self.seen_contexts)}"
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
