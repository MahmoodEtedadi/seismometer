from typing import Optional, Union

import pandas as pd
import polars as pl

from .pandas_helpers import is_valid_event


def create_metric_timeseries(
    dataframe: Union[pd.DataFrame, pl.DataFrame],
    reftime: str,
    event_col: str,
    entity_keys: list[str],
    cohort_col: str,
    *,
    time_bounds: Optional[tuple] = None,
    boolean_event: bool = False,
    censor_threshold: int = 10,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Summarize a dataframe into a frame for plotting a timeseries.

    Manipulates a dataframe with reference to

        - entity_keys - will reduce to the earliest value for each unique set of keys,
        - reftime - the time to use as the reference for the timeseries,
        - event_col - the value to use for the timeseries,
        - cohort_col - an additional metadata column for stratifying.

    Parameters
    ----------
    dataframe : Union[pd.DataFrame, pl.DataFrame]
        The input data.
    reftime : str
        The column name of the reference time.
    event_col : str
        The columns name of the value.
    entity_keys : list[str]
        A list of column names to use for summarizing the data.
    cohort_col : str
        A column containing the cohort information.
    time_bounds : Optional[tuple], optional
        An optional tuple (min, max) of inclusive times to bound the data to, by default None.
        If present the data is reduced to times within bounds prior to summarization.
        While the selection is inclusive, midnight is used if no time is passed making a maximum
        date effectively exclusive on times.
    boolean_event : bool, optional
        If True, indicates the event is boolean so negative values are filtered out, by default False.
    censor_threshold : int, optional
        The minimum number of values for a given time that are needed to not be filtered, by default 10.

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        The filtered and summarized data (same type as input).
    """
    # Detect DataFrame type
    is_polars = isinstance(dataframe, pl.DataFrame)

    # Convert to Polars if needed for processing
    if not is_polars:
        df_polars = pl.from_pandas(dataframe)
    else:
        df_polars = dataframe

    reduced = _limit_data_polars(df_polars, event_col, reftime, boolean_event=boolean_event, time_bounds=time_bounds)

    line_data = _orient_frequency_per_entity_polars(reduced, entity_keys, reftime)

    result = _censor_small_groups_polars(
        line_data, event_col, group_columns=[reftime, cohort_col], censor_threshold=censor_threshold
    )

    # Convert back to pandas if input was pandas
    if not is_polars:
        # Sort to ensure consistent ordering (pandas and polars may have different default orders)
        result_sorted = result.sort([cohort_col, event_col])
        return result_sorted.to_pandas().reset_index(drop=True)
    return result


def _orient_frequency_per_entity_polars(dataframe: pl.DataFrame, entity_keys: list[str], reftime: str) -> pl.DataFrame:
    """Reduces and aligns the data to the earliest instance of a given week."""
    # Get unique rows based on entity_keys
    line_data = dataframe.unique(subset=entity_keys, keep="first")

    # Truncate datetime to week
    # In Polars, truncate to week starts on Monday, but we'll use it as is for consistency
    line_data = line_data.with_columns(pl.col(reftime).dt.truncate("1w").alias(reftime))

    return line_data


def _limit_data_polars(
    dataframe: pl.DataFrame,
    event_col: str,
    reftime: str,
    boolean_event: bool = False,
    time_bounds: Optional[tuple] = None,
) -> pl.DataFrame:
    """Reduces the data to only include valid data points."""
    include = pl.col(reftime).is_not_null()

    if boolean_event:
        include = include & (is_valid_event(dataframe, event_col, reftime))

    reduced = dataframe.filter(include)

    if time_bounds is None:
        return reduced

    # Convert time bounds to datetime if they're strings
    min_time = pl.lit(time_bounds[0]).cast(pl.Datetime)
    max_time = pl.lit(time_bounds[1]).cast(pl.Datetime)

    return reduced.filter((pl.col(reftime) >= min_time) & (pl.col(reftime) <= max_time))


def _censor_small_groups_polars(
    dataframe: pl.DataFrame, event_col: str, group_columns: list[str], censor_threshold: int
) -> pl.DataFrame:
    """Reduces the data to only the data where each group has sufficient size per timestamp."""
    # Count occurrences in each group
    counts = dataframe.group_by(group_columns).agg(pl.col(event_col).count().alias("count"))

    # Find groups with counts <= threshold
    small_groups = counts.filter(pl.col("count") <= censor_threshold).select(group_columns)

    # Mark small groups for dropping
    small_groups = small_groups.with_columns(pl.lit(1).alias("DROPPING"))

    # Left join to mark rows from small groups
    return_data = dataframe.join(small_groups, on=group_columns, how="left")

    # Filter out rows from small groups (where DROPPING == 1)
    return_data = return_data.filter(pl.col("DROPPING").is_null())

    # Return only the relevant columns
    return return_data.select(group_columns + [event_col])
