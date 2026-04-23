from __future__ import annotations

from typing import Any

from ginjer_exercice.observability import tracing
from ginjer_exercice.schemas.ad import Ad, Brand


class FakeObservation:
    def __init__(self, name: str = "fake-observation") -> None:
        self.name = name
        self.updates: list[dict[str, Any]] = []
        self.children: list[FakeObservation] = []
        self.start_calls: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def score(self, **kwargs: Any) -> None:
        self.updates.append({"score": kwargs})

    def start_as_current_observation(self, **kwargs: Any) -> "FakeObservationContext":
        child = FakeObservation(name=kwargs.get("name", "child"))
        self.start_calls.append(kwargs)
        self.children.append(child)
        return FakeObservationContext(child)


class FakeObservationContext:
    def __init__(self, observation: FakeObservation) -> None:
        self.observation = observation

    def __enter__(self) -> FakeObservation:
        return self.observation

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.root_observation = FakeObservation(name="root")
        self.flush_called = False

    def start_as_current_observation(self, **kwargs: Any) -> FakeObservationContext:
        self.root_observation = FakeObservation(name=kwargs.get("name", "root"))
        return FakeObservationContext(self.root_observation)

    def flush(self) -> None:
        self.flush_called = True


def _sample_ad() -> Ad:
    return Ad(
        platform_ad_id="trace-test-001",
        brand=Brand.CHANEL,
        texts=[],
        media_urls=[],
    )


def test_pipeline_trace_yields_trace_context(monkeypatch: Any) -> None:
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with tracing.pipeline_trace(_sample_ad()) as trace_context:
        assert isinstance(trace_context, tracing.TraceContext)
        assert not isinstance(trace_context, FakeObservation)

    assert fake_client.flush_called is True


def test_step_span_yields_trace_context(monkeypatch: Any) -> None:
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with tracing.step_span("step_test", input_payload={"hello": "world"}) as trace_context:
        assert isinstance(trace_context, tracing.TraceContext)
        assert not isinstance(trace_context, FakeObservation)


def test_null_context_interface_matches() -> None:
    trace_methods = {
        name
        for name in dir(tracing.TraceContext)
        if not name.startswith("_") and callable(getattr(tracing.TraceContext, name))
    }
    null_methods = {
        name
        for name in dir(tracing.NullTraceContext)
        if not name.startswith("_") and callable(getattr(tracing.NullTraceContext, name))
    }

    assert trace_methods <= null_methods


def test_log_generation_accepts_all_kwargs() -> None:
    parent = FakeObservation(name="parent")
    trace_context = tracing.TraceContext(parent)

    trace_context.log_generation(
        name="llm_test",
        model="gemini-test",
        input={"prompt": "hello"},
        output={"answer": "world"},
        usage_details={"input_tokens": 10, "output_tokens": 5},
        model_parameters={"temperature": 0.3, "max_tokens": 256},
        cost_details={"input": 0.01, "output": 0.02},
        metadata={"attempt": 1},
    )

    assert len(parent.children) == 1
    child = parent.children[0]
    assert child.updates == [
        {"usage_details": {"input_tokens": 10, "output_tokens": 5}},
        {"model_parameters": {"temperature": 0.3, "max_tokens": 256}},
        {"cost_details": {"input": 0.01, "output": 0.02}},
        {"metadata": {"attempt": 1}},
    ]


def test_log_generation_with_minimal_kwargs() -> None:
    parent = FakeObservation(name="parent")
    trace_context = tracing.TraceContext(parent)

    trace_context.log_generation(
        name="llm_test",
        model="gemini-test",
        input={"prompt": "hello"},
        output={"answer": "world"},
        usage_details={"input_tokens": 10, "output_tokens": 5},
    )

    assert len(parent.children) == 1
    child = parent.children[0]
    assert child.updates == [{"usage_details": {"input_tokens": 10, "output_tokens": 5}}]


def test_null_log_generation_accepts_all_kwargs() -> None:
    trace_context = tracing.NullTraceContext()

    trace_context.log_generation(
        name="llm_test",
        model="gemini-test",
        input={"prompt": "hello"},
        output={"answer": "world"},
        usage_details={"input_tokens": 10, "output_tokens": 5},
        model_parameters={"temperature": 0.3, "max_tokens": 256},
        cost_details={"input": 0.01, "output": 0.02},
        metadata={"attempt": 1},
    )
