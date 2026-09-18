"""
Pine Script built-in symbol catalog and documentation store.

Defines the PineSymbol representation and PineScriptCatalog containing
standard Pine Script built-in functions, variables, and language keywords.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PineSymbol:
    """Represents a Pine Script language symbol with metadata and documentation."""

    name: str
    kind: str  # 'function', 'method', 'variable', 'keyword'
    signature: str
    detail: str
    documentation: str


class PineScriptCatalog:
    """Catalog of Pine Script built-in symbols, variables, and keywords."""

    def __init__(self) -> None:
        self._symbols: Dict[str, PineSymbol] = {}
        self._populate_catalog()

    def get_symbol(self, name: str) -> Optional[PineSymbol]:
        """
        Look up a symbol by exact case-sensitive name.

        Args:
            name: Fully qualified identifier or keyword name.

        Returns:
            PineSymbol if registered, otherwise None.
        """
        return self._symbols.get(name)

    def find_completions(self, prefix: str) -> List[PineSymbol]:
        """
        Find symbols matching a given prefix.

        Args:
            prefix: Symbol prefix to search for.

        Returns:
            List of matching PineSymbol objects ordered by name.
        """
        if not prefix:
            return []
        matches = [
            sym for name, sym in self._symbols.items() if name.startswith(prefix)
        ]
        return sorted(matches, key=lambda s: s.name)

    def _register(
        self,
        name: str,
        kind: str,
        signature: str,
        detail: str,
        description: str,
    ) -> None:
        """Helper to create and register a PineSymbol with Markdown docs."""
        documentation = f"```pine\n{signature}\n```\n{description}"
        symbol = PineSymbol(
            name=name,
            kind=kind,
            signature=signature,
            detail=detail,
            documentation=documentation,
        )
        self._symbols[name] = symbol

    def _populate_catalog(self) -> None:
        """Populate standard built-in functions, variables, and keywords."""
        # Technical analysis built-in functions
        self._register(
            name="ta.sma",
            kind="function",
            signature="ta.sma(source, length) -> series float",
            detail="ta.sma(source, length) -> series float",
            description="Simple moving average. Calculates the moving average of `source` for `length` bars back.",
        )
        self._register(
            name="ta.ema",
            kind="function",
            signature="ta.ema(source, length) -> series float",
            detail="ta.ema(source, length) -> series float",
            description="Exponential moving average. Calculates an exponentially weighted moving average.",
        )
        self._register(
            name="ta.rsi",
            kind="function",
            signature="ta.rsi(source, length) -> series float",
            detail="ta.rsi(source, length) -> series float",
            description="Relative Strength Index. Calculates the momentum oscillator that measures the speed and change of price movements.",
        )
        self._register(
            name="ta.stoch",
            kind="function",
            signature="ta.stoch(close, high, low, length) -> series float",
            detail="ta.stoch(close, high, low, length) -> series float",
            description="Stochastic oscillator. Calculates the %K line of the stochastic indicator.",
        )
        self._register(
            name="ta.macd",
            kind="function",
            signature="ta.macd(source, fastlen, slowlen, siglen) -> [series float, series float, series float]",
            detail="ta.macd(source, fastlen, slowlen, siglen) -> [series float, series float, series float]",
            description="Moving Average Convergence Divergence. Computes MACD line, signal line, and histogram.",
        )
        self._register(
            name="ta.crossover",
            kind="function",
            signature="ta.crossover(source1, source2) -> series bool",
            detail="ta.crossover(source1, source2) -> series bool",
            description="Returns true if `source1` crossed over `source2` on the current bar.",
        )
        self._register(
            name="ta.crossunder",
            kind="function",
            signature="ta.crossunder(source1, source2) -> series bool",
            detail="ta.crossunder(source1, source2) -> series bool",
            description="Returns true if `source1` crossed under `source2` on the current bar.",
        )
        self._register(
            name="ta.atr",
            kind="function",
            signature="ta.atr(length) -> series float",
            detail="ta.atr(length) -> series float",
            description="Average True Range. Computes volatility using Wilder's smoothing.",
        )
        self._register(
            name="ta.wma",
            kind="function",
            signature="ta.wma(source, length) -> series float",
            detail="ta.wma(source, length) -> series float",
            description="Weighted moving average of `source` for `length` bars back.",
        )
        self._register(
            name="ta.supertrend",
            kind="function",
            signature="ta.supertrend(factor, atrPeriod) -> [series float, series float]",
            detail="ta.supertrend(factor, atrPeriod) -> [series float, series float]",
            description="Supertrend indicator. Returns the supertrend value and direction.",
        )
        self._register(
            name="ta.sar",
            kind="function",
            signature="ta.sar(start, inc, max) -> series float",
            detail="ta.sar(start, inc, max) -> series float",
            description="Parabolic SAR (Stop and Reverse) indicator.",
        )
        self._register(
            name="ta.bb",
            kind="function",
            signature="ta.bb(series, length, mult) -> [series float, series float, series float]",
            detail="ta.bb(series, length, mult) -> [series float, series float, series float]",
            description="Bollinger Bands. Returns middle, upper, and lower bands.",
        )
        self._register(
            name="ta.vwap",
            kind="function",
            signature="ta.vwap(source) -> series float",
            detail="ta.vwap(source) -> series float",
            description="Volume Weighted Average Price.",
        )

        # Math built-in functions
        self._register(
            name="math.round",
            kind="function",
            signature="math.round(number, precision) -> float",
            detail="math.round(number, precision) -> float",
            description="Rounds the value to the nearest integer or specified precision.",
        )
        self._register(
            name="math.abs",
            kind="function",
            signature="math.abs(number) -> float",
            detail="math.abs(number) -> float",
            description="Returns absolute value of `number`.",
        )
        self._register(
            name="math.max",
            kind="function",
            signature="math.max(number1, number2) -> float",
            detail="math.max(number1, number2) -> float",
            description="Returns the greater of two values.",
        )
        self._register(
            name="math.min",
            kind="function",
            signature="math.min(number1, number2) -> float",
            detail="math.min(number1, number2) -> float",
            description="Returns the smaller of two values.",
        )
        self._register(
            name="math.sqrt",
            kind="function",
            signature="math.sqrt(number) -> float",
            detail="math.sqrt(number) -> float",
            description="Returns the square root of `number`.",
        )
        self._register(
            name="math.pow",
            kind="function",
            signature="math.pow(base, exponent) -> float",
            detail="math.pow(base, exponent) -> float",
            description="Raises `base` to `exponent`.",
        )

        # Plotting functions
        self._register(
            name="plot",
            kind="function",
            signature="plot(series, title, color, linewidth, style) -> plot",
            detail="plot(series, title, color, ...) -> plot",
            description="Plots a series of data on the chart.",
        )

        # Built-in variables
        self._register(
            name="close",
            kind="variable",
            signature="close: series float",
            detail="series float",
            description="Current closing price of the active bar.",
        )
        self._register(
            name="open",
            kind="variable",
            signature="open: series float",
            detail="series float",
            description="Current opening price of the active bar.",
        )
        self._register(
            name="high",
            kind="variable",
            signature="high: series float",
            detail="series float",
            description="Current high price of the active bar.",
        )
        self._register(
            name="low",
            kind="variable",
            signature="low: series float",
            detail="series float",
            description="Current low price of the active bar.",
        )
        self._register(
            name="volume",
            kind="variable",
            signature="volume: series float",
            detail="series float",
            description="Current volume of the active bar.",
        )
        self._register(
            name="time",
            kind="variable",
            signature="time: series int",
            detail="series int",
            description="Current bar UNIX timestamp in milliseconds.",
        )
        self._register(
            name="hl2",
            kind="variable",
            signature="hl2: series float",
            detail="series float",
            description="Median price: `(high + low) / 2`.",
        )
        self._register(
            name="hlc3",
            kind="variable",
            signature="hlc3: series float",
            detail="series float",
            description="Typical price: `(high + low + close) / 3`.",
        )
        self._register(
            name="ohlc4",
            kind="variable",
            signature="ohlc4: series float",
            detail="series float",
            description="Weighted average price: `(open + high + low + close) / 4`.",
        )
        self._register(
            name="bar_index",
            kind="variable",
            signature="bar_index: series int",
            detail="series int",
            description="Current bar index (zero-based counter from oldest bar).",
        )
        self._register(
            name="na",
            kind="variable",
            signature="na: simple any",
            detail="simple any",
            description="Represents an undefined or missing value in Pine Script.",
        )
        self._register(
            name="true",
            kind="variable",
            signature="true: const bool",
            detail="const bool",
            description="Boolean true constant.",
        )
        self._register(
            name="false",
            kind="variable",
            signature="false: const bool",
            detail="const bool",
            description="Boolean false constant.",
        )

        # Built-in keywords
        self._register(
            name="strategy",
            kind="keyword",
            signature="strategy(title, shorttitle, overlay, ...)",
            detail="strategy declaration",
            description="Declaration statement specifying the script is an automated trading strategy.",
        )
        self._register(
            name="indicator",
            kind="keyword",
            signature="indicator(title, shorttitle, overlay, ...)",
            detail="indicator declaration",
            description="Declaration statement specifying the script is a chart indicator.",
        )
        self._register(
            name="var",
            kind="keyword",
            signature="var type var_name = init_value",
            detail="var declaration",
            description="Initializes a variable once on the first bar; preserves state across bars.",
        )
        self._register(
            name="varip",
            kind="keyword",
            signature="varip type var_name = init_value",
            detail="varip declaration",
            description="Initializes a variable once and preserves values between intrabar ticks.",
        )
        self._register(
            name="if",
            kind="keyword",
            signature="if condition\n    expression",
            detail="if statement",
            description="Conditional branch execution.",
        )
        self._register(
            name="else",
            kind="keyword",
            signature="else\n    expression",
            detail="else statement",
            description="Alternative branch for conditional statements.",
        )
        self._register(
            name="for",
            kind="keyword",
            signature="for counter = start to end [by step]\n    expression",
            detail="for loop",
            description="Loop statement executing a block across a counter range.",
        )
        self._register(
            name="while",
            kind="keyword",
            signature="while condition\n    expression",
            detail="while loop",
            description="Loop statement repeatedly executing while condition remains true.",
        )
        self._register(
            name="switch",
            kind="keyword",
            signature="switch [expression]\n    value => statement",
            detail="switch statement",
            description="Multi-way branching and pattern matching statement.",
        )