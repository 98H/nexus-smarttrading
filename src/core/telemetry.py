from typing import Any, Dict, Optional

from opentelemetry import context, propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# OpenTelemetry enforces that TracerProvider can only be set once per process.
# In test environments with isolated TracerProviders per test, allow updating
# the global tracer provider by resetting the once flag.
if getattr(trace.set_tracer_provider, "__name__", "") != "_set_tracer_provider":
    _original_set_tracer_provider = trace.set_tracer_provider

    def _set_tracer_provider(tracer_provider: trace.TracerProvider) -> None:
        if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
            try:
                trace._TRACER_PROVIDER_SET_ONCE._done = False
            except Exception:
                pass
        _original_set_tracer_provider(tracer_provider)
        try:
            trace._TRACER_PROVIDER = tracer_provider
        except Exception:
            pass

    trace.set_tracer_provider = _set_tracer_provider


def setup_telemetry(
    service_name: str,
    environment: str = "production",
) -> TracerProvider:
    """Initialize OpenTelemetry TracerProvider, resource, and global W3C propagators."""
    if not isinstance(service_name, str):
        raise TypeError(f"service_name must be a str, got {type(service_name).__name__}")
    if environment is not None and not isinstance(environment, str):
        raise TypeError(f"environment must be a str, got {type(environment).__name__}")

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": environment,
        }
    )
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    propagate.set_global_textmap(TraceContextTextMapPropagator())
    return provider


def get_tracer(name: str, version: Optional[str] = None) -> trace.Tracer:
    """Retrieve an OpenTelemetry tracer from the currently configured TracerProvider."""
    return trace.get_tracer(name, version)


def inject_trace_context(
    carrier: Dict[str, str],
    ctx: Optional[context.Context] = None,
) -> None:
    """Inject current or provided W3C trace context into a carrier dictionary."""
    TraceContextTextMapPropagator().inject(carrier, context=ctx)


def extract_trace_context(
    carrier: Dict[str, Any],
    ctx: Optional[context.Context] = None,
) -> context.Context:
    """Extract W3C trace context from a carrier dictionary."""
    try:
        normalized_carrier = (
            {str(k).lower(): v for k, v in carrier.items()} if carrier else {}
        )
        return TraceContextTextMapPropagator().extract(normalized_carrier, context=ctx)
    except Exception:
        return ctx or context.get_current()