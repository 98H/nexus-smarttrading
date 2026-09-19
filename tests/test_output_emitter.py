import math
from typing import Any
import pytest

from src.runtime.output_emitter import OutputEmitter


# ============================================================================
# Helpers
# ============================================================================

def _normalize_series_dict(collection: Any, key: str) -> dict[str, Any]:
    """Helper to extract a series from dictionary-based or list-based visual collections."""
    if isinstance(collection, dict):
        if key in collection:
            return collection[key]
        raise KeyError(f"Series '{key}' not found in output dictionary.")
    if isinstance(collection, list):
        for item in collection:
            if isinstance(item, dict) and item.get("title") == key:
                return item
    raise KeyError(f"Series '{key}' not found in output list.")


def _normalize_markers_list(collection: Any, title: str | None = None) -> list[dict[str, Any]]:
    """Helper to flatten markers from either list or grouped dictionary format."""
    if isinstance(collection, list):
        if title is not None:
            return [m for m in collection if m.get("title") == title]
        return collection
    if isinstance(collection, dict):
        if title is not None:
            val = collection.get(title, [])
            return val if isinstance(val, list) else [val]
        flat: list[dict[str, Any]] = []
        for val in collection.values():
            if isinstance(val, list):
                flat.extend(val)
            elif isinstance(val, dict):
                flat.append(val)
        return flat
    return []


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def emitter() -> OutputEmitter:
    """Provides a clean OutputEmitter instance for testing."""
    return OutputEmitter()


# ============================================================================
# AC 1: emit_plot Tests
# ============================================================================

def test_emit_plot_records_metadata_and_time_indexed_values(emitter: OutputEmitter):
    """Given series values and styling parameters, emit_plot records title, style, color, and values."""
    emitter.set_bar(bar_index=0, time=1672531200)
    emitter.emit_plot(
        title="SMA 20",
        value=105.5,
        color="#2196F3",
        style="line",
        linewidth=2,
        offset=0,
    )

    outputs = emitter.get_outputs()
    assert "plots" in outputs
    plot_series = _normalize_series_dict(outputs["plots"], "SMA 20")

    assert plot_series["title"] == "SMA 20"
    assert plot_series["color"] == "#2196F3"
    assert plot_series["style"] == "line"
    assert plot_series["linewidth"] == 2
    assert plot_series["offset"] == 0

    values = plot_series["values"]
    assert len(values) == 1
    assert values[0]["bar_index"] == 0
    assert values[0]["time"] == 1672531200
    assert values[0]["value"] == 105.5


def test_emit_plot_records_multiple_sequential_bars(emitter: OutputEmitter):
    """emit_plot captures sequential bar values with correct time-indexed points."""
    bars = [
        (0, 1000, 10.0),
        (1, 2000, 11.5),
        (2, 3000, 12.8),
    ]
    for bar_idx, t, val in bars:
        emitter.set_bar(bar_index=bar_idx, time=t)
        emitter.emit_plot(title="EMA", value=val, color="orange")

    outputs = emitter.get_outputs()
    series = _normalize_series_dict(outputs["plots"], "EMA")
    values = series["values"]

    assert len(values) == 3
    for idx, (bar_idx, t, val) in enumerate(bars):
        assert values[idx]["bar_index"] == bar_idx
        assert values[idx]["time"] == t
        assert values[idx]["value"] == val


def test_emit_plot_multiple_distinct_series(emitter: OutputEmitter):
    """emit_plot isolates separate series titles without crosstalk."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plot(title="Fast", value=100.0, color="blue")
    emitter.emit_plot(title="Slow", value=95.0, color="red")

    outputs = emitter.get_outputs()
    plots = outputs["plots"]

    fast = _normalize_series_dict(plots, "Fast")
    slow = _normalize_series_dict(plots, "Slow")

    assert fast["values"][0]["value"] == 100.0
    assert slow["values"][0]["value"] == 95.0


def test_emit_plot_records_none_and_nan_values(emitter: OutputEmitter):
    """emit_plot records None or NaN values to represent gaps in series."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plot(title="GapPlot", value=None)
    emitter.set_bar(bar_index=1, time=2000)
    emitter.emit_plot(title="GapPlot", value=math.nan)

    series = _normalize_series_dict(emitter.get_outputs()["plots"], "GapPlot")
    values = series["values"]

    assert values[0]["value"] is None
    assert math.isnan(values[1]["value"])


def test_emit_plot_dynamic_color_per_bar(emitter: OutputEmitter):
    """emit_plot records dynamic bar-level colors when color varies across bars."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plot(title="Trend", value=1.0, color="green")
    emitter.set_bar(bar_index=1, time=2000)
    emitter.emit_plot(title="Trend", value=-1.0, color="red")

    values = _normalize_series_dict(emitter.get_outputs()["plots"], "Trend")["values"]
    assert values[0]["color"] == "green"
    assert values[1]["color"] == "red"


def test_emit_plot_explicit_time_and_index_kwargs(emitter: OutputEmitter):
    """Explicit bar_index and time kwargs override any active bar context."""
    emitter.set_bar(bar_index=10, time=10000)
    emitter.emit_plot(title="Override", value=42.0, bar_index=99, time=99999)

    values = _normalize_series_dict(emitter.get_outputs()["plots"], "Override")["values"]
    assert values[0]["bar_index"] == 99
    assert values[0]["time"] == 99999


def test_emit_plot_invalid_linewidth_raises_value_error(emitter: OutputEmitter):
    """emit_plot raises ValueError when linewidth is <= 0."""
    emitter.set_bar(bar_index=0, time=1000)
    with pytest.raises(ValueError):
        emitter.emit_plot(title="BadWidth", value=1.0, linewidth=0)
    with pytest.raises(ValueError):
        emitter.emit_plot(title="BadWidth", value=1.0, linewidth=-2)


# ============================================================================
# AC 2: emit_plotshape and emit_plotchar Tests
# ============================================================================

def test_emit_plotshape_captures_marker_when_condition_true(emitter: OutputEmitter):
    """emit_plotshape records marker attributes when condition evaluates to True."""
    emitter.set_bar(bar_index=3, time=1672534800)
    emitter.emit_plotshape(
        condition=True,
        title="Long Signal",
        style="shape_triangleup",
        location="belowbar",
        color="#00FF00",
        offset=-1,
        text="BUY",
        textcolor="#FFFFFF",
        size="small",
    )

    outputs = emitter.get_outputs()
    assert "plotshapes" in outputs
    markers = _normalize_markers_list(outputs["plotshapes"], title="Long Signal")

    assert len(markers) == 1
    marker = markers[0]
    assert marker["title"] == "Long Signal"
    assert marker["style"] == "shape_triangleup"
    assert marker["location"] == "belowbar"
    assert marker["color"] == "#00FF00"
    assert marker["offset"] == -1
    assert marker["text"] == "BUY"
    assert marker["textcolor"] == "#FFFFFF"
    assert marker["size"] == "small"
    assert marker["bar_index"] == 3
    assert marker["time"] == 1672534800


def test_emit_plotshape_ignores_falsy_conditions(emitter: OutputEmitter):
    """emit_plotshape does not capture visual markers when condition is False or None."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plotshape(condition=False, title="False Signal", style="shape_circle")
    emitter.set_bar(bar_index=1, time=2000)
    emitter.emit_plotshape(condition=None, title="None Signal", style="shape_circle")

    markers = _normalize_markers_list(emitter.get_outputs()["plotshapes"])
    assert len(markers) == 0


def test_emit_plotshape_default_styling_attributes(emitter: OutputEmitter):
    """emit_plotshape applies sensible default styling when optional params are omitted."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plotshape(condition=True)

    markers = _normalize_markers_list(emitter.get_outputs()["plotshapes"])
    assert len(markers) == 1
    marker = markers[0]
    assert marker["style"] == "shape_circle"
    assert marker["location"] == "abovebar"
    assert marker["offset"] == 0


def test_emit_plotshape_invalid_location_raises_value_error(emitter: OutputEmitter):
    """emit_plotshape raises ValueError if an unsupported location is specified."""
    emitter.set_bar(bar_index=0, time=1000)
    with pytest.raises(ValueError):
        emitter.emit_plotshape(condition=True, location="invalid_position")


def test_emit_plotchar_captures_marker_with_char_and_attributes(emitter: OutputEmitter):
    """emit_plotchar records marker with char, location, offset, and styling."""
    emitter.set_bar(bar_index=5, time=1672542000)
    emitter.emit_plotchar(
        condition=True,
        char="★",
        title="Star Marker",
        location="top",
        color="#FFD700",
        offset=2,
        text="ALERT",
        textcolor="#000000",
        size="large",
    )

    outputs = emitter.get_outputs()
    assert "plotchars" in outputs
    markers = _normalize_markers_list(outputs["plotchars"], title="Star Marker")

    assert len(markers) == 1
    marker = markers[0]
    assert marker["char"] == "★"
    assert marker["title"] == "Star Marker"
    assert marker["location"] == "top"
    assert marker["color"] == "#FFD700"
    assert marker["offset"] == 2
    assert marker["text"] == "ALERT"
    assert marker["textcolor"] == "#000000"
    assert marker["size"] == "large"
    assert marker["bar_index"] == 5
    assert marker["time"] == 1672542000


def test_emit_plotchar_ignores_falsy_conditions(emitter: OutputEmitter):
    """emit_plotchar does not emit active marker when condition is False, 0, or None."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plotchar(condition=False, char="X")
    emitter.set_bar(bar_index=1, time=2000)
    emitter.emit_plotchar(condition=0, char="Y")
    emitter.set_bar(bar_index=2, time=3000)
    emitter.emit_plotchar(condition=None, char="Z")

    markers = _normalize_markers_list(emitter.get_outputs()["plotchars"])
    assert len(markers) == 0


def test_emit_plotchar_empty_char_raises_value_error(emitter: OutputEmitter):
    """emit_plotchar raises ValueError if char is an empty string."""
    emitter.set_bar(bar_index=0, time=1000)
    with pytest.raises(ValueError):
        emitter.emit_plotchar(condition=True, char="")


def test_emit_plotchar_invalid_location_raises_value_error(emitter: OutputEmitter):
    """emit_plotchar raises ValueError for unrecognized location strings."""
    emitter.set_bar(bar_index=0, time=1000)
    with pytest.raises(ValueError):
        emitter.emit_plotchar(condition=True, char="A", location="underground")


# ============================================================================
# AC 3: emit_plotcandle Tests
# ============================================================================

def test_emit_plotcandle_records_ohlc_and_colors(emitter: OutputEmitter):
    """emit_plotcandle records candle points with OHLC values and respective border/wick colors."""
    emitter.set_bar(bar_index=0, time=1672531200)
    emitter.emit_plotcandle(
        open=100.0,
        high=108.5,
        low=97.0,
        close=106.0,
        title="Custom Candle",
        color="#26A69A",
        wickcolor="#80CBC4",
        bordercolor="#004D40",
        offset=0,
    )

    outputs = emitter.get_outputs()
    assert "plotcandles" in outputs
    candles = _normalize_markers_list(outputs["plotcandles"], title="Custom Candle")

    assert len(candles) == 1
    candle = candles[0]
    assert candle["open"] == 100.0
    assert candle["high"] == 108.5
    assert candle["low"] == 97.0
    assert candle["close"] == 106.0
    assert candle["color"] == "#26A69A"
    assert candle["wickcolor"] == "#80CBC4"
    assert candle["bordercolor"] == "#004D40"
    assert candle["offset"] == 0
    assert candle["bar_index"] == 0
    assert candle["time"] == 1672531200


def test_emit_plotcandle_multi_bar_sequence(emitter: OutputEmitter):
    """emit_plotcandle records sequence of candles across evaluation bars."""
    bars = [
        (0, 1000, 10.0, 15.0, 9.0, 14.0, "green", "green", "green"),
        (1, 2000, 14.0, 16.0, 11.0, 12.0, "red", "red", "red"),
        (2, 3000, 12.0, 13.5, 11.8, 13.0, "blue", "gray", "black"),
    ]
    for b_idx, t, o, h, l, c, col, wcol, bcol in bars:
        emitter.set_bar(bar_index=b_idx, time=t)
        emitter.emit_plotcandle(
            open=o,
            high=h,
            low=l,
            close=c,
            color=col,
            wickcolor=wcol,
            bordercolor=bcol,
        )

    candles = _normalize_markers_list(emitter.get_outputs()["plotcandles"])
    assert len(candles) == 3
    for idx, (b_idx, t, o, h, l, c, col, wcol, bcol) in enumerate(bars):
        assert candles[idx]["bar_index"] == b_idx
        assert candles[idx]["time"] == t
        assert candles[idx]["open"] == o
        assert candles[idx]["high"] == h
        assert candles[idx]["low"] == l
        assert candles[idx]["close"] == c
        assert candles[idx]["color"] == col
        assert candles[idx]["wickcolor"] == wcol
        assert candles[idx]["bordercolor"] == bcol


def test_emit_plotcandle_high_less_than_low_raises_value_error(emitter: OutputEmitter):
    """emit_plotcandle raises ValueError when high is strictly less than low."""
    emitter.set_bar(bar_index=0, time=1000)
    with pytest.raises(ValueError):
        emitter.emit_plotcandle(open=100.0, high=95.0, low=99.0, close=97.0)


# ============================================================================
# AC 4: Consolidated get_outputs & Lifecycle Tests
# ============================================================================

def test_initial_outputs_empty_payload(emitter: OutputEmitter):
    """A fresh emitter produces an empty, categorized payload without errors."""
    outputs = emitter.get_outputs()
    assert isinstance(outputs, dict)
    for category in ("plots", "plotshapes", "plotchars", "plotcandles"):
        assert category in outputs
        assert len(outputs[category]) == 0


def test_get_outputs_consolidated_payload_across_evaluation_bars(emitter: OutputEmitter):
    """get_outputs aggregates plot, plotshape, plotchar, and plotcandle across bars."""
    # Bar 0: plot + candle
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plot(title="SMA", value=100.0)
    emitter.emit_plotcandle(open=99.0, high=101.0, low=98.0, close=100.0)

    # Bar 1: plot + plotshape + candle
    emitter.set_bar(bar_index=1, time=2000)
    emitter.emit_plot(title="SMA", value=102.0)
    emitter.emit_plotshape(condition=True, title="Breakout", style="shape_triangleup")
    emitter.emit_plotcandle(open=100.0, high=103.0, low=99.5, close=102.5)

    # Bar 2: plot + plotchar + candle
    emitter.set_bar(bar_index=2, time=3000)
    emitter.emit_plot(title="SMA", value=101.5)
    emitter.emit_plotchar(condition=True, char="!", title="Warning")
    emitter.emit_plotcandle(open=102.5, high=104.0, low=101.0, close=101.5)

    payload = emitter.get_outputs()
    assert isinstance(payload, dict)

    # Validate categories presence
    assert set(payload.keys()) >= {"plots", "plotshapes", "plotchars", "plotcandles"}

    # Validate plots
    plot_series = _normalize_series_dict(payload["plots"], "SMA")
    assert len(plot_series["values"]) == 3

    # Validate shapes
    shapes = _normalize_markers_list(payload["plotshapes"])
    assert len(shapes) == 1
    assert shapes[0]["title"] == "Breakout"

    # Validate chars
    chars = _normalize_markers_list(payload["plotchars"])
    assert len(chars) == 1
    assert chars[0]["char"] == "!"

    # Validate candles
    candles = _normalize_markers_list(payload["plotcandles"])
    assert len(candles) == 3


def test_clear_resets_recorded_events(emitter: OutputEmitter):
    """clear() purges all recorded visual series back to empty baseline."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plot(title="P", value=1.0)
    emitter.emit_plotshape(condition=True, title="S")
    emitter.emit_plotchar(condition=True, char="C", title="C")
    emitter.emit_plotcandle(open=1.0, high=2.0, low=0.5, close=1.5)

    emitter.clear()

    fresh_outputs = emitter.get_outputs()
    assert len(fresh_outputs["plots"]) == 0
    assert len(_normalize_markers_list(fresh_outputs["plotshapes"])) == 0
    assert len(_normalize_markers_list(fresh_outputs["plotchars"])) == 0
    assert len(_normalize_markers_list(fresh_outputs["plotcandles"])) == 0


def test_emit_without_bar_context_or_explicit_time_raises_value_error():
    """Emitting events without active bar context or explicit time/index raises ValueError."""
    fresh_emitter = OutputEmitter()
    with pytest.raises(ValueError):
        fresh_emitter.emit_plot(title="OrphanPlot", value=10.0)

    with pytest.raises(ValueError):
        fresh_emitter.emit_plotshape(condition=True)

    with pytest.raises(ValueError):
        fresh_emitter.emit_plotchar(condition=True, char="*")

    with pytest.raises(ValueError):
        fresh_emitter.emit_plotcandle(open=1.0, high=2.0, low=0.5, close=1.5)


def test_get_outputs_immutability_and_snapshots(emitter: OutputEmitter):
    """Modifying returned output payload or emitting subsequent bars does not mutate prior snapshot."""
    emitter.set_bar(bar_index=0, time=1000)
    emitter.emit_plot(title="SnapshotTest", value=100.0)

    snapshot_1 = emitter.get_outputs()
    series_1 = _normalize_series_dict(snapshot_1["plots"], "SnapshotTest")
    assert len(series_1["values"]) == 1

    emitter.set_bar(bar_index=1, time=2000)
    emitter.emit_plot(title="SnapshotTest", value=200.0)

    snapshot_2 = emitter.get_outputs()
    series_2 = _normalize_series_dict(snapshot_2["plots"], "SnapshotTest")
    assert len(series_2["values"]) == 2

    # Snapshot 1 remains unaffected
    assert len(series_1["values"]) == 1