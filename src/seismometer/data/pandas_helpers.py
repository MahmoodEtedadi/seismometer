import logging
import warnings
from numbers import Number
from typing import Optional, get_args

import pandas as pd  # Only for backward compatibility with pandas DataFrames in _ensure_polars() and type checking
import polars as pl

from seismometer.configuration import ConfigurationError
from seismometer.configuration.model import MergeStrategies

logger = logging.getLogger("seismometer")

MAXIMUM_COUNT_CATS = 15


def _ensure_polars(df) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars if needed (for test compatibility)."""
    if not isinstance(df, pl.DataFrame) and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def merge_windowed_event(
    predictions: pl.DataFrame,
    predtime_col: str,
    events: pl.DataFrame,
    event_label: str,
    pks: list[str],
    *,
    min_leadtime_hrs: Number = 0,
    window_hrs: Optional[Number] = None,
    event_base_val_col: str = "Value",
    event_base_time_col: str = "Time",
    event_base_val_dtype: str = "float",
    sort: bool = True,
    merge_strategy: str = "forward",
    impute_val_with_time: Optional[Number | str] = 1,
    impute_val_no_time: Optional[Number | str] = 0,
) -> pl.DataFrame:
    """
    Merges a single windowed event into a predictions dataframe

    Adds two new event columns: a _Value column with the event value and a _Time column with the event time.
    Ground-truth labeling for a model is considered an event and can have a time associated with it.

    Joins on a set of keys and associates the first event occurring after the prediction time.  The following special
    cases are also applied:

    Early predictions drop timing - if a prediction occurs before all recorded events of the type, the label is kept
    for analyses but the time is removed.
    Imputation of no event to negative label - if no row in the events frame is present for the prediction keys, it is
    assumed to be a Negative label (default 0) but will not have an event time.


    Parameters
    ----------
    predictions : pl.DataFrame
        The predictions or features frame where each row represents a prediction.
    predtime_col : str
        The column in the predictions frame indicating the timestamp when inference occurred.
    events : pl.DataFrame
        The narrow events dataframe
    event_label : str
        The category name of the event to merge, expected to be a value in events.Type.
    pks : list[str]
        A list of primary keys on which to perform the merge, keys are column names occurring in both predictions and
        events dataframes.
    min_leadtime_hrs : Number, optional
        The number of hour offset to be required for prediction, by default 0.
    window_hrs : Optional[Number], optional
        The number of hours the window of predictions of interest should be limited to, by default None.
        If None, then all predictions occurring before a known event will be included.
        If used with min_leadtime_hrs, the entire window is shifted maintaining its size. The maximum lookback for a
        prediction is window_hrs + min_leadtime_hrs.
    event_base_val_col : str, optional
        The name of the column in the events frame to merge as the _Value, by default 'Value'.
    event_base_val_dtype : str
        The data type to cast the event value column to, by default 'float'.
    event_base_time_col : str, optional
        The name of the column in the events frame to merge as the _Time, by default 'Time'.
    sort : bool
        Whether or not to sort the predictions/events dataframes, by default True.
    merge_strategy : str
        The method to use when merging the event data, by default 'forward'.
        Options are 'forward', 'nearest', 'first', 'last', and 'count'.
        See seismometer.configuration.model for more information.
    impute_val_with_time : Optional[Number|str], optional
        The value to impute for the label if timestamp exists, defaults to 1.
    impute_val_no_time : Optional[Number|str], optional
        The value to impute for the label if no timestamp exists, defaults to 0.

    Returns
    -------
    pl.DataFrame
        The predictions dataframe with the new time and value columns for the event specified.

    Raises
    ------
    ValueError
        At least one column in pks must be in both the predictions and events dataframes.
    """
    # Handle test compatibility - detect input type and return same type
    input_is_pandas = isinstance(predictions, pd.DataFrame)
    original_predictions_pandas = predictions if input_is_pandas else None
    predictions = _ensure_polars(predictions)
    events = _ensure_polars(events)

    # Validate merge strategy
    if merge_strategy not in get_args(MergeStrategies):
        raise ValueError(
            f"Invalid merge strategy {merge_strategy} for {event_label}."
            + f" Must be one of: {', '.join(get_args(MergeStrategies))}."
        )

    # Validate and resolve
    r_ref = "~~reftime~~"
    pks = [
        col for col in pks if col in events.columns and col in predictions.columns
    ]  # Ensure existence in both frames
    if len(pks) == 0:
        raise ValueError("No common keys found between predictions and events.")

    min_offset = pl.duration(hours=min_leadtime_hrs)

    if sort:
        predictions = predictions.sort(predtime_col, maintain_order=True)
        events = events.sort(event_base_time_col, maintain_order=True)

    # Preprocess events : reduce and rename
    one_event = _one_event(events, event_label, event_base_val_col, event_base_time_col, pks)
    if len(one_event) == 0:
        logger.debug(f"No events found for {event_label} with keys {pks}.")
        if input_is_pandas:
            # Simulate pandas inplace=True behavior by updating original DataFrame
            result_pandas = predictions.to_pandas()
            # Update existing columns (sorting may have changed order)
            for col in result_pandas.columns:
                original_predictions_pandas[col] = result_pandas[col].values
            return original_predictions_pandas
        return predictions

    event_time_col = event_time(event_label)
    event_val_col = event_value(event_label)
    # Add r_ref column (only if event_time_col is datetime type)
    if one_event.schema[event_time_col] in [pl.Datetime, pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]:
        one_event = one_event.with_columns((pl.col(event_time_col) - min_offset).alias(r_ref))
    else:
        # If not datetime (e.g., all nulls), just copy the time column
        one_event = one_event.with_columns(pl.col(event_time_col).alias(r_ref))

    # When merging counts we want to apply the windowing BEFORE the merge.
    # So this case is handled separately due to needing some additional arguments.
    if merge_strategy == "count":
        # Immediately return the merged frame with the counts to avoid unnecessary processing.
        logger.debug(f"Merging event counts for {event_label} with columns {pks}.")
        result = _merge_event_counts(
            predictions,
            one_event,
            pks,
            event_label,
            event_val_col,
            window_hrs=window_hrs,
            min_offset=min_offset,
            l_ref=predtime_col,
            r_ref=r_ref,
        )
        # Convert back to pandas if input was pandas
        if input_is_pandas:
            result_pandas = result.to_pandas() if not isinstance(result, pd.DataFrame) else result
            # Update existing columns and add new ones
            for col in result_pandas.columns:
                original_predictions_pandas[col] = result_pandas[col].values
            return original_predictions_pandas
        return result

    # merge event specified by merge_strategy for each prediction
    event_ref = event_time_col if merge_strategy in ["forward", "nearest"] else r_ref
    logger.debug(
        f"Merging {event_label} with columns {pks} using strategy {merge_strategy} on {predtime_col} and {event_ref}."
    )
    predictions = _merge_with_strategy(
        predictions,
        one_event,
        pks,
        pred_ref=predtime_col,
        event_ref=event_ref,
        event_display=event_label,
        merge_strategy=merge_strategy,
    )

    # Note that filtering happens after merging.
    logger.debug(f"Starting post-processing of events for {event_label}. There are {len(predictions)} predictions.")
    if window_hrs is not None:  # Clear out events outside window
        logger.debug(f"Filtering out events outside of the {window_hrs} hour window.")
        max_lookback = pl.duration(hours=window_hrs)  # r_ref has already been moved by min_offset.
        filter_condition = (pl.col(predtime_col) < (pl.col(r_ref) - max_lookback)) | (
            pl.col(predtime_col) > pl.col(r_ref)
        )
        # Count how many will be filtered before applying
        changed_count = predictions.select(filter_condition.sum()).item()
        # Apply filter by setting values to null
        predictions = predictions.with_columns(
            [
                pl.when(filter_condition).then(None).otherwise(pl.col(event_val_col)).alias(event_val_col),
                pl.when(filter_condition).then(None).otherwise(pl.col(event_time_col)).alias(event_time_col),
            ]
        )
        logger.debug(f"Filtered out {changed_count} event values for {event_label} outside the window.")

    predictions = post_process_event(
        predictions,
        event_val_col,
        event_time_col,
        column_dtype=event_base_val_dtype,
        impute_val_with_time=impute_val_with_time,
        impute_val_no_time=impute_val_no_time,
    )
    event_val_col = event_value(event_label)
    if logger.isEnabledFor(logging.INFO) and event_val_col in predictions.columns:
        added_events_count = predictions.select((pl.col(event_val_col) == 1).sum()).item()
        logger.info(
            f"Kept {added_events_count} events (value=1) for {event_label} after applying specified lookback window "
            f"of {window_hrs} hours and offset of {min_leadtime_hrs}."
        )

    predictions = predictions.drop(r_ref)
    # Return pandas if input was pandas (for test compatibility)
    if input_is_pandas:
        # Simulate pandas inplace=True behavior by updating original DataFrame
        result_pandas = predictions.to_pandas()
        # Update existing columns and add new ones
        for col in result_pandas.columns:
            original_predictions_pandas[col] = result_pandas[col].values
        return original_predictions_pandas
    return predictions


def _one_event(
    events: pl.DataFrame, event_label: str, event_base_val_col: str, event_base_time_col: str, pks: list[str]
) -> pl.DataFrame:
    """Reduces the events dataframe to those rows associated with the event_label, preemptively renaming to the
    columns to what a join should use and reducing columns to pks + event value and time."""
    # Handle test compatibility - detect input type and return same type
    input_is_pandas = isinstance(events, pd.DataFrame)
    events = _ensure_polars(events)
    expected_columns = pks + [event_base_val_col, event_base_time_col]
    one_event = events.filter(pl.col("Type") == event_label).select(expected_columns)
    one_event = one_event.rename(
        {event_base_time_col: event_time(event_label), event_base_val_col: event_value(event_label)}
    )
    # Return pandas if input was pandas (for test compatibility)
    return one_event.to_pandas() if input_is_pandas else one_event


def post_process_event(
    dataframe: pl.DataFrame,
    label_col: str,
    time_col: str,
    *,
    column_dtype: str = "float",
    impute_val_with_time: Optional[Number | str] = 1,
    impute_val_no_time: Optional[Number | str] = 0,
) -> pl.DataFrame:
    """
    Infers and casts events.

    Default assumptions are for binary classifications (cast as float to maximize compatibility with analyses).
    A row that does not have any documentation of an event defaults to a negative (0) label - impute_val_no_time.
    A row that has a timestamp but no event value defaults to a positive (1) label - impute_val_with_time.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The dataframe to modify.
    label_col : str
        The column specifying the value to infer.
    time_col : time_col
        The time column associated with the value to infer.
    column_dtype : str
        The data type to cast the label column to, done after imputation; by default 'float'.
    impute_val_with_time : Optional[Number|str], optional
        The value to impute for the label if timestamp exist, defaults to 1.
    impute_val_no_time : Optional[Number|str], optional
        The value to impute for the label if no timestamp exist, defaults to 0.

    Returns
    -------
    pl.DataFrame
        The dataframe with potentially modified labels.
    """
    # Handle test compatibility - detect input type and return same type
    input_is_pandas = isinstance(dataframe, pd.DataFrame)
    dataframe = _ensure_polars(dataframe)

    logger.debug(f"Post-processing events for {label_col} and {time_col}.")
    if label_col not in dataframe.columns or time_col not in dataframe.columns:
        logger.debug(f"Columns {label_col} or {time_col} not found in dataframe, skipping post-processing for events.")
        return dataframe.to_pandas() if input_is_pandas else dataframe

    # Store original null state for logging
    label_was_null = pl.col(label_col).is_null()
    time_is_not_null = pl.col(time_col).is_not_null()

    # Use polars for imputation - handle Nones
    impute_with = impute_val_with_time if impute_val_with_time is not None else 1
    impute_without = impute_val_no_time if impute_val_no_time is not None else 0

    # Apply imputation logic using with_columns
    if impute_val_with_time is not None or impute_val_no_time is not None:
        dataframe = dataframe.with_columns(
            pl.when(label_was_null & time_is_not_null)
            .then(pl.lit(impute_with))
            .when(label_was_null)
            .then(pl.lit(impute_without))
            .otherwise(pl.col(label_col))
            .alias(label_col)
        )

    if column_dtype is None:
        return dataframe.to_pandas() if input_is_pandas else dataframe

    # cast after imputation - supports nonnullable types
    dataframe = try_casting(dataframe, label_col, column_dtype)

    # Log how many rows were imputed/changed
    imputed_with_time = dataframe.select(
        (label_was_null & time_is_not_null & pl.col(label_col).is_not_null()).sum()
    ).item()
    imputed_no_time = dataframe.select(pl.col(label_col).is_null().sum()).item()
    logger.debug(
        f"Post-processing of events for {label_col} and {time_col} complete. "
        f"Imputed {imputed_with_time} rows with time, {imputed_no_time} rows with no time."
    )

    # Return pandas if input was pandas (for test compatibility)
    return dataframe.to_pandas() if input_is_pandas else dataframe


def try_casting(dataframe: pl.DataFrame, column: str, column_dtype: str) -> pl.DataFrame:
    """
    Attempts to cast a column to a specified data type.

    For pandas DataFrames, modifies in place for backward compatibility.
    For polars DataFrames, returns a new DataFrame.

    Will convert the specified column to a data type, raising a ConfigurationError if the conversion fails.
    Does multistep casts to get strings "1.0" into format int 1.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The dataframe to modify.
    column : str
        The column to cast.
    column_dtype : str
        The data type to cast the column to.

    Returns
    -------
    pl.DataFrame
        The dataframe with the casted column (new DataFrame for polars, modified in-place for pandas).

    Raises
    ------
    ConfigurationError
        If the column cannot be cast to the specified data type.
    """
    # Handle pandas DataFrames directly with pandas operations
    if isinstance(dataframe, pd.DataFrame):
        try:
            if "int" in column_dtype.lower():  # "1.0" -> 1.0 then 1.0 -> 1
                dataframe[column] = dataframe[column].astype(float)
            dataframe[column] = dataframe[column].astype(column_dtype)
            return dataframe
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(
                f"Cannot cast '{event_name(column)}' values to '{column_dtype}'. "
                + "Update dictionary config or contact the model owner."
            ) from exc

    # Handle polars DataFrames
    try:
        # Convert string dtype to polars dtype
        dtype_map = {
            "float": pl.Float64,
            "float64": pl.Float64,
            "float32": pl.Float32,
            "int": pl.Int64,
            "int64": pl.Int64,
            "int32": pl.Int32,
            "int16": pl.Int16,
            "int8": pl.Int8,
            "string": pl.String,
            "str": pl.String,
            "object": pl.String,
            "category": pl.Categorical,
            "bool": pl.Boolean,
            "datetime64[ns]": pl.Datetime("ns"),
            "datetime64": pl.Datetime("ns"),
            "datetime": pl.Datetime("ns"),
            "Int64": pl.Int64,  # Pandas nullable Int64
        }
        polars_dtype = (
            dtype_map.get(column_dtype.lower(), column_dtype) if isinstance(column_dtype, str) else column_dtype
        )

        if isinstance(column_dtype, str) and "int" in column_dtype.lower():  # "1.0" -> 1.0 then 1.0 -> 1
            dataframe = dataframe.with_columns(pl.col(column).cast(pl.Float64))
        dataframe = dataframe.with_columns(pl.col(column).cast(polars_dtype))

        return dataframe
    except (ValueError, TypeError, pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as exc:
        raise ConfigurationError(
            f"Cannot cast '{event_name(column)}' values to '{column_dtype}'. "
            + "Update dictionary config or contact the model owner."
        ) from exc


def _merge_event_counts(
    left: pl.DataFrame,
    right: pl.DataFrame,
    pks: list[str],
    event_name: str,
    event_label: str,
    window_hrs: Optional[Number] = None,
    min_offset: pl.Duration = pl.duration(hours=0),
    l_ref: str = "Time",
    r_ref: str = "~~reftime~~",
) -> pl.DataFrame:
    """Creates a new column for each event in the right frame's event_label column,
    counting the number of times that event has occurred"""
    # Handle test compatibility - detect input type and return same type
    input_is_pandas = isinstance(left, pd.DataFrame)
    left = _ensure_polars(left)
    right = _ensure_polars(right)
    logger.debug(f"Merging event counts for {event_name} with columns {pks}.")

    # Validate and convert min_offset type
    if isinstance(min_offset, pd.Timedelta):
        # Convert pandas Timedelta to polars Duration
        min_offset = pl.duration(microseconds=min_offset.total_seconds() * 1_000_000)
    elif not isinstance(min_offset, (pl.Duration, pl.Expr)):
        raise TypeError(f"min_offset must be a Timedelta or Duration, not {type(min_offset).__name__}")

    if l_ref == r_ref:
        raise ValueError(
            f"`l_ref` and `r_ref` must be different to avoid column collisions during merge (both are '{l_ref}')."
        )

    if window_hrs is not None:
        # Filter out rows with missing times if checking window hours
        right_filtered = right.filter(pl.col(r_ref).is_not_null())
        if len(right_filtered) == 0:
            logger.warning(f"No times found for {event_name}! Unable to merge any counts.")
            return left.to_pandas() if input_is_pandas else left
        if (diff := len(right) - len(right_filtered)) > 0:
            logger.warning(f"Found {diff} rows with missing times for {event_name}. These rows will be ignored.")
            right = right_filtered

        max_lookback = pl.duration(hours=window_hrs) + min_offset  # Keep window the specified size
        right = right.join(left.select(pks + [l_ref]), on=pks, how="left")
        right = right.filter(
            pl.col(l_ref) <= pl.col(r_ref)
        )  # Filter to only events that happened at or after the prediction
        right = right.filter(
            pl.col(l_ref) > (pl.col(r_ref) - max_lookback)
        )  # Filter to only events that happened within the window

    # Validate number of categories to create columns for
    pop_counts = right.group_by(event_label).agg(pl.len().alias("count")).sort("count", descending=True)
    if (N := len(pop_counts)) > MAXIMUM_COUNT_CATS:
        logger.warning(
            f"Maximum number of unique events to count is {MAXIMUM_COUNT_CATS}, but {N} were found for {event_name}. "
            + f"Only the top {MAXIMUM_COUNT_CATS} by number of appearances will be included."
        )
        # Filter right frame to the only contain the top MAXIMUM_COUNT_CATS events
        events_to_count = pop_counts.head(MAXIMUM_COUNT_CATS).select(event_label).to_series()
        right = right.filter(pl.col(event_label).is_in(events_to_count))

    event_name_map = {
        str(event): event_value_count(str(event_label), str(event))
        for event in right.select(event_label).unique().to_series()
    }  # Create dictionary to map column names with - ensure keys are strings for polars rename

    # Create a value counts dataframe where each event is a column containing the count of that
    # event grouped by the primary keys.
    val_counts = (
        right.group_by(pks + [event_label])
        .agg(pl.len().alias("count"))
        .pivot(index=pks, on=event_label, values="count")
        .fill_null(0)
        .rename(event_name_map)
    )

    left = left.join(val_counts, on=pks, how="left")  # Merge counts into left frame
    count_cols = list(event_name_map.values())
    # Fill any missing counts for rows that didn't have any events
    left = left.with_columns([pl.col(col).fill_null(0) for col in count_cols])

    # Return pandas if input was pandas (for test compatibility)
    return left.to_pandas() if input_is_pandas else left


def _merge_with_strategy(
    predictions: pl.DataFrame,
    one_event: pl.DataFrame,
    pks: list[str],
    *,
    pred_ref: str = "Time",
    event_ref: str = "Time",
    event_display: str = "an event",
    merge_strategy: str = "forward",
) -> pl.DataFrame:
    """
    Merges the right frame into the left based on a set of exact match primary keys and merge strategy.

    Parameters
    ----------
    predictions : pl.DataFrame
        The left frame, usually of predictions. Assumed to be sorted by time.
    one_event : pl.DataFrame
        The right frame to merge, assumed to be of events. Assumed to be sorted by time if applicable.
    pks : list[str]
        The list of columns to require exact matches during the merge.
    pred_ref : str, optional
        The column in the left (prediction) frame to use as a reference point in the distance match, by default 'Time'.
    event_ref : str, optional
        The column in the right (event) frame to use reference in the distance match, by default 'Time'.
    event_display : str, optional
        The name of the event to display in warning messages, by default "an event"
    merge_strategy : str
        The method to use when merging the event data, by default 'forward'.

    Returns
    -------
    pl.DataFrame
        The merged dataframe.
    """
    # Handle test compatibility - detect input type and return same type
    input_is_pandas = isinstance(predictions, pd.DataFrame)
    predictions = _ensure_polars(predictions)
    one_event = _ensure_polars(one_event)
    try:
        ct_times = one_event.select(pl.col(event_ref).is_not_null().sum()).item()

        # If there are no times in the event frame, merge the first row for each group
        if ct_times == 0:
            # Set the filtered frame to the first row for each group and throw a value error
            # which is passed before merging.
            one_event_filtered = one_event.unique(subset=pks, keep="first")
            raise ValueError(f"No times found for {event_display}, merging first row for each group.")

        if ct_times != len(one_event):
            logger.warning(f"Inconsistent event times for {event_display}, only considering events with times.")
            one_event = one_event.filter(pl.col(event_ref).is_not_null())

        if merge_strategy == "forward" or merge_strategy == "nearest":
            # Ensure both frames are sorted for join_asof
            one_event_filtered = one_event.filter(pl.col(event_ref).is_not_null())
            # Suppress polars warning about sortedness when 'by' groups are used
            # We know the data is sorted (sorted at beginning of merge_windowed_event)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Sortedness of columns cannot be checked when 'by' groups provided"
                )
                predictions_with_events = predictions.join_asof(
                    one_event_filtered,
                    left_on=pred_ref,
                    right_on=event_ref,
                    by=pks,
                    strategy=merge_strategy,
                )
            event_val_col = event_value(event_display)
            if logger.isEnabledFor(logging.INFO) and event_val_col in predictions_with_events.columns:
                added_events_count = predictions_with_events.select((pl.col(event_val_col) == 1).sum()).item()
                logger.info(f"Added {added_events_count} events (value=1) for {event_display}.")

            logger.debug(
                f"Merged {event_display} using {merge_strategy} strategy on {pred_ref} and {event_ref} with "
                f"keys {pks}. There are {len(predictions_with_events)} predictions."
            )
            # Return pandas if input was pandas (for test compatibility)
            return predictions_with_events.to_pandas() if input_is_pandas else predictions_with_events

        # Assume sorted on event_ref before being passed in
        if merge_strategy == "first":
            logger.debug(f"Updating events to only keep the first occurrence for each {event_display}.")
            one_event_filtered = one_event.unique(subset=pks, keep="first")
        if merge_strategy == "last":
            logger.debug(f"Updating events to only keep the last occurrence for each {event_display}.")
            one_event_filtered = one_event.unique(subset=pks, keep="last")

    except ValueError as e:
        logger.warning(e)
        pass

    result = predictions.join(one_event_filtered, on=pks, how="left")
    # Return pandas if input was pandas (for test compatibility)
    return result.to_pandas() if input_is_pandas else result


def max_aggregation(df: pl.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pl.DataFrame:
    """
    Aggregates the DataFrame by selecting the maximum score value.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pl.DataFrame
        The aggregated DataFrame.
    """
    df = _ensure_polars(df)  # Test compatibility
    if ref_event is None:
        raise ValueError("With aggregation_method 'max', ref_event is required.")

    event_val = event_value(ref_event)
    ref_score = _resolve_score_col(df, score)
    df = df.sort([event_val, ref_score], descending=True)
    return df.unique(subset=pks, keep="first")


def min_aggregation(df: pl.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pl.DataFrame:
    """
    Aggregates the DataFrame by selecting the minimum score value.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pl.DataFrame
        The aggregated DataFrame.
    """
    df = _ensure_polars(df)  # Test compatibility
    if ref_event is None:
        raise ValueError("With aggregation_method 'min', ref_event is required.")

    event_val = event_value(ref_event)
    ref_score = _resolve_score_col(df, score)
    df = df.sort([event_val, ref_score], descending=[True, False])
    return df.unique(subset=pks, keep="first")


def first_aggregation(df: pl.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pl.DataFrame:
    """
    Aggregates the DataFrame by selecting the first occurrence based on event time.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pl.DataFrame
        The aggregated DataFrame.
    """
    df = _ensure_polars(df)  # Test compatibility
    if ref_time is None:
        raise ValueError("With aggregation_method 'first', ref_time is required.")

    reference_time = _resolve_time_col(df, ref_time)
    df = df.filter(pl.col(reference_time).is_not_null())
    df = df.sort(reference_time)
    return df.unique(subset=pks, keep="first")


def last_aggregation(df: pl.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pl.DataFrame:
    """
    Aggregates the DataFrame by selecting the last occurrence based on event time.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pl.DataFrame
        The aggregated DataFrame.
    """
    df = _ensure_polars(df)  # Test compatibility
    if ref_time is None:
        raise ValueError("With aggregation_method 'last', ref_time is required.")

    reference_time = _resolve_time_col(df, ref_time)
    df = df.filter(pl.col(reference_time).is_not_null())
    df = df.sort(reference_time, descending=True)
    return df.unique(subset=pks, keep="first")


def event_score(
    merged_frame: pl.DataFrame,
    pks: list[str],
    score: str,
    ref_time: Optional[str] = None,
    ref_event: Optional[str] = None,
    aggregation_method: str = "max",
) -> pl.DataFrame:
    """
    Reduces a dataframe of all predictions to a single row of significance; such as the max or most recent value for
    an entity.
    Supports max/min for value only scores, and last/first if a reference timestamp is provided.

    Parameters
    ----------
    merged_frame : pl.DataFrame
        The dataframe with score and event data, such as those having an event added via merge_windowed_event.
    pks : list[str]
        A list of identifying keys on which to aggregate, such as Id.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
        Required when aggregation_method requires a time reference (e.g., 'first', 'last').
        Note that we drop NaT rows first and consequently we pick the row satisfying the
        aggregation_method that also corresponds to a positive case for the associated event.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.
        Required when aggregation_method requires an event reference to prioritize positive cases (e.g., 'max', 'min')
        Note that we pick the row satisfying the aggregation_method among scores associated with a positive case of
        ref_event if there are any positive cases. In case there are no positive case, we just pick the row satisfying
        the aggregation_method.
    aggregation_method : str, optional
        A string describing the method to select a value, by default 'max'.

    Returns
    -------
    pl.DataFrame
        The reduced dataframe with one row per combination of pks.
    """
    logger.debug(
        f"Combining scores using {aggregation_method} for {score} on ref_time: {ref_time} "
        + f"and ref_event: {ref_event}"
    )
    # Handle test compatibility - detect input type and return same type
    input_is_pandas = isinstance(merged_frame, pd.DataFrame)

    pks = [c for c in pks if c in merged_frame.columns]

    aggregation_methods = {
        "max": max_aggregation,
        "min": min_aggregation,
        "first": first_aggregation,
        "last": last_aggregation,
    }

    if aggregation_method not in aggregation_methods:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    df = aggregation_methods[aggregation_method](merged_frame, pks, score, ref_time, ref_event)

    # Return pandas if input was pandas (for test compatibility)
    if input_is_pandas:
        return df.to_pandas()
    return df


def get_model_scores(
    dataframe: pl.DataFrame,
    entity_keys: list[str],
    score_col: str,
    ref_time: Optional[str],
    ref_event: Optional[str],
    aggregation_method: str = "max",
    per_context_id: bool = False,
) -> pl.DataFrame:
    """
    Reduces a dataframe of all predictions to a single row of significance; such as the max or most recent value for
    an entity.
    Supports max/min for value only scores, and last/first if a reference timestamp is provided.

    Parameters
    ----------
    merged_frame : pl.DataFrame
        The dataframe with score and event data, such as those having an event added via merge_windowed_event.
    entity_keys : list[str]
        A list of identifying keys on which to aggregate, such as Id.
    score_col : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.
    aggregation_method : str, optional
        A string describing the method to select a value, by default 'max'.
    per_context_id : bool, optional
        If True, limits data to one row per context_id, by default False.

    Returns
    -------
    pl.DataFrame
        The reduced dataframe with one row per combination of pks.
    """
    if per_context_id:
        # event_score now handles pandas/polars compatibility internally
        return event_score(
            dataframe,
            entity_keys,
            score=score_col,
            ref_time=ref_time,
            ref_event=ref_event,
            aggregation_method=aggregation_method,
        )
    return dataframe


# region Core Methods
def event_value(event: str) -> str:
    """Converts an event name into the value column name."""
    if event.endswith("_Value"):
        return event

    if event.endswith("_Time"):
        event = event[:-5]
    return f"{event}_Value"


def event_time(event: str) -> str:
    """Converts an event name into the time column name."""
    if event.endswith("_Time"):
        return event

    if event.endswith("_Value"):
        event = event[:-6]
    return f"{event}_Time"


def event_value_count(event_label: str, event_value: str) -> str:
    """Converts a value of an event column into the count column name."""
    event_label = event_name(event_label)

    if event_value.endswith("_Count"):
        return f"{event_label}~{event_value}"

    return f"{event_label}~{event_value}_Count"


def event_name(event: str) -> str:
    """Converts an event column name into the the event name."""
    if event.endswith("_Time"):
        return event[:-5]

    if event.endswith("_Value"):
        return event[:-6]
    return event


def event_value_name(event_value: str) -> str:
    """Converts event value count column into the event value name."""
    val = event_value
    if "~" in val:
        val = val.split("~")[1]
    if val.endswith("_Count"):
        val = val[:-6]

    return val


def is_valid_event(dataframe, event: str, ref: str):
    """
    Creates a mask excluding rows (False) where the event occurs before the reference time.
    If the comparison cannot be made, all rows will be considered valid (True).

    Returns a Polars expression if input is Polars, pandas Series if input is pandas.
    """
    import polars as pl

    # Handle pandas DataFrames
    if isinstance(dataframe, pd.DataFrame):
        if event_time(event) not in dataframe.columns or ref not in dataframe.columns:
            return pd.Series([True] * len(dataframe), index=dataframe.index)
        return dataframe[ref] <= dataframe[event_time(event)]

    # Handle Polars DataFrames - return expression
    if event_time(event) not in dataframe.columns or ref not in dataframe.columns:
        return pl.lit(True)
    return pl.col(ref) <= pl.col(event_time(event))


def _resolve_time_col(dataframe: pl.DataFrame, ref_event: str) -> str:
    """
    Determines the time column to use based on existence in the dataframe.
    First assumes it is an event, and checks the time column associated with that name.
    Defaults to the ref_event being the exact column.
    """
    if ref_event is None:
        raise ValueError("Reference event must be specified for last/first summarization")
    ref_time = event_time(ref_event)
    if ref_time not in dataframe.columns:
        if ref_event not in dataframe.columns:
            raise ValueError(f"Reference time column {ref_time} not found in dataframe")
        ref_time = ref_event
    return ref_time


def _resolve_score_col(dataframe: pl.DataFrame, score: str) -> str:
    """
    Determines the value column to use based on existence in the dataframe.
    First assumes the score is a column.
    Defaults to the ref_event being the exact column.
    """
    if score not in dataframe.columns:
        if event_value(score) not in dataframe.columns:
            raise ValueError(f"Score column {score} not found in dataframe.")
        score = event_value(score)
    return score


def analytics_metric_name(metric_names: list[str], existing_metric_starts: list[str], column_name: str) -> str:
    """In the analytics table, often the provided column name is not the actual
    metric name that we want to log. Here, we extract the desired metric name.

    Parameters
    ----------
    metric_names : list[str]
        What metrics already exist.
    existing_metric_values : list[str]
        What strings can start the mangled column name.
    column_name : str
        The name of the column we are trying to make into a metric.

    Returns
    -------
    str
        The resulting metric name, or none if no match was found.
    """
    if column_name in metric_names:
        return column_name
    else:
        for value in existing_metric_starts:
            if column_name.startswith(f"{value}_"):
                return column_name.lstrip(f"{value}_")
    return None


# endregion
