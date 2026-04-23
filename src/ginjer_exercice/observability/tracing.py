import logging
from contextlib import contextmanager
from typing import Any, ContextManager, Generator

from pydantic import BaseModel

from ..schemas.ad import Ad
from .client import get_langfuse_client

logger = logging.getLogger(__name__)


def _safe_model_dump(obj: Any) -> Any:
    """Serialize Pydantic values recursively before sending them to Langfuse."""
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _safe_model_dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_model_dump(item) for item in obj]
    return obj


def _log_sdk_warning(action: str, exc: Exception) -> None:
    """Log a non-fatal warning when the Langfuse SDK raises."""
    logger.warning("Langfuse SDK error during %s: %s", action, exc)


class TraceContext:
    """Wrap a Langfuse observation so raw SDK objects never leave observability."""

    def __init__(self, span: Any) -> None:
        self._span = span

    @property
    def trace_id(self) -> str | None:
        return getattr(self._span, "trace_id", None) or getattr(self._span, "id", None)

    def update_output(self, output: Any) -> None:
        """Update the current observation output.

        Use this after a pipeline step has produced its final result and the span
        should expose that structured payload in Langfuse. The payload is always
        serialized through the local safe serializer before reaching the SDK.

        Args:
            output: Final step payload to attach to the current observation.

        Returns:
            None.

        Raises:
            None. SDK errors are caught, logged as warnings, and suppressed.
        """
        try:
            self._span.update(output=_safe_model_dump(output))
        except Exception as exc:
            _log_sdk_warning("update_output", exc)

    def update_metadata(self, **kwargs: Any) -> None:
        """Attach metadata fields to the current observation.

        Use this for incremental diagnostic fields that should remain on the span
        without changing the main output payload. Keyword values are serialized
        recursively before being forwarded to Langfuse as metadata.

        Args:
            **kwargs: Metadata fields to add to the observation.

        Returns:
            None.

        Raises:
            None. SDK errors are caught, logged as warnings, and suppressed.
        """
        try:
            self._span.update(metadata=_safe_model_dump(kwargs))
        except Exception as exc:
            _log_sdk_warning("update_metadata", exc)

    def log_generation(
        self,
        name: str,
        model: str,
        input: Any,
        output: Any,
        usage_details: dict[str, int],
        model_parameters: dict[str, Any] | None = None,
        cost_details: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a nested Langfuse generation under the current OTEL context.

        Use this immediately after a successful LLM call to record the request,
        response, usage, and optional generation attributes on a child span.
        Nesting is delegated to Langfuse OTEL context propagation via
        ``start_as_current_observation``.

        Args:
            name: Generation observation name.
            model: Model identifier reported to Langfuse.
            input: Serialized LLM input payload.
            output: Serialized LLM output payload.
            usage_details: Usage counters to publish on the generation.
            model_parameters: Optional model invocation parameters.
            cost_details: Optional generation cost details.
            metadata: Optional generation metadata payload.

        Returns:
            None.

        Raises:
            None. SDK errors are caught, logged as warnings, and suppressed.
        """
        try:
            with self._span.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=_safe_model_dump(input),
                output=_safe_model_dump(output),
            ) as generation:
                if usage_details is not None:
                    generation.update(usage_details=_safe_model_dump(usage_details))
                if model_parameters is not None:
                    generation.update(model_parameters=_safe_model_dump(model_parameters))
                if cost_details is not None:
                    generation.update(cost_details=_safe_model_dump(cost_details))
                if metadata is not None:
                    generation.update(metadata=_safe_model_dump(metadata))
        except Exception as exc:
            _log_sdk_warning("log_generation", exc)

    @contextmanager
    def child_span(
        self,
        name: str,
        input_payload: Any = None,
    ) -> ContextManager["TraceContext"]:
        """Open a child span nested under the current observation.

        Use this to create a scoped child trace context for a sub-operation while
        keeping raw Langfuse observations hidden from the caller. The yielded
        object is always another ``TraceContext`` wrapper.

        Args:
            name: Child span name shown in Langfuse.
            input_payload: Optional serialized input attached to the child span.

        Returns:
            A context manager yielding a nested ``TraceContext``.

        Raises:
            None. SDK errors are caught, logged as warnings, and suppressed.
        """
        try:
            with self._span.start_as_current_observation(
                as_type="span",
                name=name,
                input=_safe_model_dump(input_payload),
            ) as child:
                yield TraceContext(child)
        except Exception as exc:
            _log_sdk_warning("child_span", exc)
            yield NullTraceContext()

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score to the current observation.

        Use this for deterministic evaluations or post-step quality signals that
        should appear directly on the current observation in Langfuse. The score
        is best-effort and must never affect pipeline execution.

        Args:
            name: Score name to create in Langfuse.
            value: Numeric score value.
            comment: Optional human-readable score comment.

        Returns:
            None.

        Raises:
            None. SDK errors are caught, logged as warnings, and suppressed.
        """
        try:
            self._span.score(name=name, value=value, comment=comment)
        except Exception as exc:
            _log_sdk_warning("score", exc)


class NullTraceContext:
    """No-op implementation matching the public ``TraceContext`` interface."""

    @property
    def trace_id(self) -> None:
        return None

    def update_output(self, output: Any) -> None:
        """Ignore output updates when observability is disabled.

        Use this null object in code paths that should stay branch-free even when
        Langfuse is unavailable. The call is intentionally ignored.

        Args:
            output: Unused output payload.

        Returns:
            None.

        Raises:
            None. This null object never raises.
        """
        return None

    def update_metadata(self, **kwargs: Any) -> None:
        """Ignore metadata updates when observability is disabled.

        Use this null object to preserve the production API without introducing
        conditional checks in pipeline code. All metadata fields are discarded.

        Args:
            **kwargs: Unused metadata fields.

        Returns:
            None.

        Raises:
            None. This null object never raises.
        """
        return None

    def log_generation(
        self,
        name: str,
        model: str,
        input: Any,
        output: Any,
        usage_details: dict[str, int],
        model_parameters: dict[str, Any] | None = None,
        cost_details: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Ignore generation logging when observability is disabled.

        Use this null object so LLM providers can call the tracing API without
        caring whether Langfuse is configured. All arguments are ignored.

        Args:
            name: Unused generation name.
            model: Unused model name.
            input: Unused generation input.
            output: Unused generation output.
            usage_details: Unused usage counters.
            model_parameters: Unused model invocation parameters.
            cost_details: Unused generation cost details.
            metadata: Unused metadata payload.

        Returns:
            None.

        Raises:
            None. This null object never raises.
        """
        return None

    @contextmanager
    def child_span(
        self,
        name: str,
        input_payload: Any = None,
    ) -> ContextManager["NullTraceContext"]:
        """Yield another no-op trace context for nested spans.

        Use this to preserve the nested context-manager shape of production code
        when observability is disabled. The yielded context stays fully inert.

        Args:
            name: Unused child span name.
            input_payload: Unused child span input.

        Returns:
            A context manager yielding ``NullTraceContext``.

        Raises:
            None. This null object never raises.
        """
        yield NullTraceContext()

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """Ignore score creation when observability is disabled.

        Use this null object to keep score publication best-effort and completely
        decoupled from pipeline success. All score arguments are ignored.

        Args:
            name: Unused score name.
            value: Unused score value.
            comment: Unused score comment.

        Returns:
            None.

        Raises:
            None. This null object never raises.
        """
        return None


@contextmanager
def pipeline_trace(ad: Ad, session_id: str | None = None) -> Generator[TraceContext, None, None]:
    """Open the root trace context for a single advertisement pipeline run."""
    langfuse = get_langfuse_client()
    if langfuse is None:
        yield NullTraceContext()
        return

    with langfuse.start_as_current_observation(
        as_type="span",
        name=f"pipeline_{ad.platform_ad_id}",
        input=_safe_model_dump(ad),
    ) as root_span:
        tags = [ad.brand.value]
        metadata = {
            "platform_ad_id": ad.platform_ad_id,
            "media_count": len(ad.media_urls),
            "text_count": len(ad.texts),
        }
        root_span.update(
            session_id=session_id,
            user_id=ad.brand.value,
            tags=tags,
            metadata=_safe_model_dump(metadata),
        )
        try:
            yield TraceContext(root_span)
        finally:
            langfuse_client = get_langfuse_client()
            if langfuse_client is not None:
                langfuse_client.flush()


@contextmanager
def step_span(name: str, input_payload: Any = None) -> Generator[TraceContext, None, None]:
    """Open a child step span and yield its wrapped trace context."""
    langfuse = get_langfuse_client()
    if langfuse is None:
        yield NullTraceContext()
        return

    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input=_safe_model_dump(input_payload),
    ) as span:
        yield TraceContext(span)
