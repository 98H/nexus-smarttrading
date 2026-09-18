"""Multi-Timeframe Series Pull (request.security) Engine.

Provides functionality to align higher-timeframe series onto a lower-timeframe
primary series index using closed-bar forward filling without lookahead bias.
"""

from __future__ import annotations

from typing import TypeVar, Union, overload

import pandas as pd

SeriesOrDataFrame = TypeVar("SeriesOrDataFrame", pd.Series, pd.DataFrame)


def _resolve_frequency_offset(index: pd.DatetimeIndex) -> pd.DateOffset:
    """Resolve and validate the timeframe frequency offset of a secondary series index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The secondary series DatetimeIndex to validate.

    Returns
    -------
    pd.DateOffset
        The parsed timeframe offset representing bar duration.

    Raises
    ------
    ValueError
        If the index timestamps are irregular, unaligned, or do not match a valid grid.
    """
    if len(index) < 2 and index.freq is None:
        raise ValueError("Secondary series must contain at least 2 timestamps to determine frequency.")

    if len(index) >= 2:
        deltas = index[1:] - index[:-1]
        if not (deltas == deltas[0]).all():
            raise ValueError(
                "Secondary series has irregular or unaligned timestamps that do not match a valid timeframe grid."
            )

    freq_str: str | None = None
    if index.freq is not None:
        freq_str = index.freqstr or str(index.freq)
    else:
        freq_str = pd.infer_freq(index)

    if freq_str is None:
        raise ValueError(
            "Secondary series has irregular or unaligned timestamps that do not match a valid timeframe grid."
        )

    try:
        return pd.tseries.frequencies.to_offset(freq_str)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse timeframe frequency offset '{freq_str}': {exc}"
        ) from exc


def _validate_inputs(
    primary_series: Union[pd.Series, pd.DataFrame],
    secondary_series: Union[pd.Series, pd.DataFrame],
) -> None:
    """Validate index types, emptiness, ordering, and timezone alignment.

    Parameters
    ----------
    primary_series : Union[pd.Series, pd.DataFrame]
        Primary lower-timeframe series or dataframe.
    secondary_series : Union[pd.Series, pd.DataFrame]
        Secondary higher-timeframe series or dataframe.

    Raises
    ------
    TypeError
        If inputs are not pandas Series or DataFrame instances.
    ValueError
        If inputs are empty, non-chronological, contain duplicates, or have mismatched timezones.
    """
    if not isinstance(primary_series, (pd.Series, pd.DataFrame)):
        raise TypeError("primary_series must be a pandas Series or DataFrame.")
    if not isinstance(secondary_series, (pd.Series, pd.DataFrame)):
        raise TypeError("secondary_series must be a pandas Series or DataFrame.")

    if primary_series.empty:
        raise ValueError("Primary series cannot be empty.")
    if secondary_series.empty:
        raise ValueError("Secondary series cannot be empty.")

    if not isinstance(primary_series.index, pd.DatetimeIndex):
        raise ValueError("Primary series index must be a DatetimeIndex.")
    if not isinstance(secondary_series.index, pd.DatetimeIndex):
        raise ValueError("Secondary series index must be a DatetimeIndex.")

    if primary_series.index.tz != secondary_series.index.tz:
        raise ValueError(
            f"Timezone mismatch: primary series timezone is '{primary_series.index.tz}', "
            f"secondary series timezone is '{secondary_series.index.tz}'."
        )

    if not primary_series.index.is_monotonic_increasing:
        raise ValueError("Primary series index must be strictly chronological.")
    if primary_series.index.has_duplicates:
        raise ValueError("Primary series index must not contain duplicate timestamps.")

    if not secondary_series.index.is_monotonic_increasing:
        raise ValueError("Secondary series index must be strictly chronological.")
    if secondary_series.index.has_duplicates:
        raise ValueError("Secondary series index must not contain duplicate timestamps.")


@overload
def resolve_security_series(
    primary_series: Union[pd.Series, pd.DataFrame],
    secondary_series: pd.Series,
    lookahead: bool = False,
) -> pd.Series:
    ...


@overload
def resolve_security_series(
    primary_series: Union[pd.Series, pd.DataFrame],
    secondary_series: pd.DataFrame,
    lookahead: bool = False,
) -> pd.DataFrame:
    ...


def resolve_security_series(
    primary_series: Union[pd.Series, pd.DataFrame],
    secondary_series: SeriesOrDataFrame,
    lookahead: bool = False,
) -> SeriesOrDataFrame:
    """Align higher-timeframe secondary data onto a lower-timeframe primary index.

    When `lookahead=False`, closed-bar alignment is enforced by indexing secondary
    values at their bar close times (open time + bar duration offset), forward-filling
    onto the primary index so that no future data leaks before a higher-timeframe
    bar has finalized.

    Parameters
    ----------
    primary_series : Union[pd.Series, pd.DataFrame]
        Primary lower-timeframe series or dataframe whose index is strictly preserved.
    secondary_series : SeriesOrDataFrame
        Secondary higher-timeframe series or dataframe to be aligned.
    lookahead : bool, default False
        If False, values are shifted to bar close timestamps to prevent lookahead bias.
        If True, values are available at bar open timestamps.

    Returns
    -------
    SeriesOrDataFrame
        Aligned secondary series or dataframe sharing the exact index of `primary_series`.

    Raises
    ------
    ValueError
        If either series is empty, non-chronological, has duplicate or unaligned timestamps,
        or if timezones do not match.
    """
    _validate_inputs(primary_series, secondary_series)
    offset = _resolve_frequency_offset(secondary_series.index)

    sec = secondary_series.copy()
    if not lookahead:
        sec.index = secondary_series.index + offset

    combined_index = primary_series.index.union(sec.index)
    aligned = sec.reindex(combined_index).ffill().reindex(primary_series.index)
    aligned.index = primary_series.index

    if isinstance(secondary_series, pd.Series) and isinstance(aligned, pd.Series):
        aligned.name = secondary_series.name

    return aligned