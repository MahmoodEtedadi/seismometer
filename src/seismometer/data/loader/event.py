import logging

import pandas as pd

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Event

from .pipeline import ConfigOnlyHook

logger = logging.getLogger("seismometer")


def get_data_loader(config: ConfigProvider) -> ConfigOnlyHook:
    """
    Returns the proper data loader function from the event file extension.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    ConfigOnlyHook
        The predictions dataframe.
    """
    loaders = {
        ".csv": csv_loader,
        ".tsv": tsv_loader,
        ".parquet": parquet_loader,
    }
    return loaders.get(config.event_path.suffix.lower(), parquet_loader)


def csv_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Loads the events frame from a csv file based on config.event_path.

    Maps the non-key columns to the standard names "Type", "Time", "Value".
    Will log debug message and return an empty dataframe if read_csv fails.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    pd.DataFrame
        the events dataframe with standarized column names.
    """
    return _sv_loader(config, ",")


def tsv_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Loads the events frame from a csv file based on config.event_path.

    Maps the non-key columns to the standard names "Type", "Time", "Value".
    Will log debug message and return an empty dataframe if read_csv fails.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    pd.DataFrame
        the events dataframe with standarized column names.
    """
    return _sv_loader(config, "\t")


def parquet_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Loads the events frame from a parquet file based on config.event_path.

    Maps the non-key columns to the standard names "Type", "Time", "Value".
    Will log debug message and return an empty dataframe if read_parquet fails.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    pd.DataFrame
        the events dataframe with standarized column names.
    """
    import polars as pl

    logger.debug(f"Loading events from {config.event_path}.")
    try:
        events = pl.read_parquet(config.event_path).rename(
            {config.ev_type: "Type", config.ev_time: "Time", config.ev_value: "Value"}
        )
        logger.debug(f"Loaded {len(events)} events from {config.event_path}.")
    except BaseException:
        logger.warning(f"No events found at {config.event_path}")
        events = pl.DataFrame({col: [] for col in config.entity_keys + ["Type", "Time", "Value"]})

    return events


def post_transform_fn(config: ConfigProvider, events):
    """
    Converts the Time column in events to a datetime64[ns] type, to be compatible with other operations.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.
    events : Union[pd.DataFrame, pl.DataFrame]
        The events dataframe.

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        The transformed events dataframe.
    """
    import polars as pl

    # Time column in events is known
    if isinstance(events, pl.DataFrame):
        # Polars: cast to datetime with ns precision
        events = events.with_columns(pl.col("Time").cast(pl.Datetime("ns")))
    else:
        # Pandas
        events["Time"] = events["Time"].astype("<M8[ns]")

    logger.debug("Transformed events dataframe with Time as datetime64[ns].")
    return events


# data merges
def merge_onto_predictions(config: ConfigProvider, event_frame: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merges each configured event onto the predictions dataframe.

    Will create event columns (_Value + _Time) for each configured event.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.
    event_frame : pd.DataFrame
        The events dataframe.
    dataframe : pd.DataFrame
        The target predictions dataframe.

    Returns
    -------
    pd.DataFrame
        The merged dataframe of predictions plus new event columns.
    """
    import polars as pl

    logger.debug("Merging events onto predictions dataframe.")

    # Prepare predictions
    if isinstance(dataframe, pl.DataFrame):
        dataframe = (
            dataframe.sort(config.predict_time, maintain_order=True)
            .unique(subset=config.entity_keys + [config.predict_time])
            .drop_nulls(subset=[config.predict_time])
        )
    else:
        dataframe = (
            dataframe.sort_values(config.predict_time, kind="mergesort")
            .drop_duplicates(subset=config.entity_keys + [config.predict_time])
            .dropna(subset=[config.predict_time])
        )
    logger.debug(
        f"Sorted predictions dataframe by {config.predict_time}, dropping duplicate (entity time, prediction time) "
        "and dropping predictions with Null prediction time."
    )

    # Prepare events
    if isinstance(event_frame, pl.DataFrame):
        event_frame = event_frame.sort("Time", maintain_order=True)
    else:
        event_frame = event_frame.sort_values("Time", kind="mergesort")

    # Merge each event sequentially
    for one_event in config.events.values():
        logger.debug(f"Processing event {one_event.display_name} with sources {one_event.source}")
        # Get dtype
        event_dtype = _get_source_type(config, one_event)

        # Merge
        if one_event.window_hr:
            logger.debug(
                f"Windowing event {one_event.display_name} to lookback {one_event.window_hr} "
                + f"offset by {one_event.offset_hr}"
            )
            dataframe = _merge_event(
                config,
                one_event.source,
                dataframe,
                event_frame,
                event_dtype=event_dtype,
                window_hrs=one_event.window_hr,
                offset_hrs=one_event.offset_hr,
                display=one_event.display_name,
                sort=False,
                impute_pos=one_event.impute_val,
            )
        else:  # No lookback
            logger.debug(f"Merging event {one_event.display_name}")
            dataframe = _merge_event(
                config,
                one_event.source,
                dataframe,
                event_frame,
                event_dtype=event_dtype,
                display=one_event.display_name,
                sort=False,
                impute_pos=one_event.impute_val,
            )

        # Impute no event
        if one_event.impute_val and one_event.impute_val != 0:
            event_val = pdh.event_value(one_event.display_name)
            logger.warning(
                f"Event {one_event.display_name} specified impute; "
                + "currently missing event value is being inferred based on timestamp existence."
            )
            impute = one_event.impute_val
            dataframe[event_val] = dataframe[event_val].fillna(impute)

    return dataframe


def _merge_event(
    config,
    event_cols,
    dataframe,
    event_frame,
    event_dtype=None,
    offset_hrs=0,
    window_hrs=None,
    display="",
    sort=True,
    impute_pos=1,
    impute_neg=0,
) -> pd.DataFrame:
    """Wrapper for calling merge_windowed_event with the correct event column names."""
    import polars as pl

    disp_event = display if display else event_cols[0]
    logger.debug(f"Merging event {disp_event} with columns {event_cols} onto predictions dataframe.")

    # Replace Type column values
    if isinstance(event_frame, pl.DataFrame):
        # Polars version - replace values in Type column where Type is in event_cols
        event_frame_filtered = event_frame.with_columns(
            pl.when(pl.col("Type").is_in(event_cols)).then(pl.lit(disp_event)).otherwise(pl.col("Type")).alias("Type")
        )
    else:
        # Pandas version
        event_frame_filtered = event_frame.replace({"Type": event_cols}, disp_event)

    return pdh.merge_windowed_event(
        dataframe,
        config.predict_time,
        event_frame_filtered,
        disp_event,
        config.entity_keys,
        min_leadtime_hrs=offset_hrs,
        window_hrs=window_hrs,
        event_base_time_col="Time",
        event_base_val_dtype=event_dtype,
        sort=sort,
        merge_strategy=config.events[disp_event].merge_strategy,
        impute_val_with_time=impute_pos,
        impute_val_no_time=impute_neg,
    )


def _get_source_type(config: ConfigProvider, event: Event) -> str:
    """Get the dtype of the first source of the specified event."""
    dtype = None
    multitypes = set()  # Aggregate woarning messages
    for source in event.source:
        new_type = None
        if (source_defn := config.event_defs.get(source, None)) is not None:
            new_type = source_defn.dtype

        if new_type is None:
            continue
        if dtype is None:
            dtype = new_type
            continue

        if dtype != new_type:
            multitypes.add(new_type)

    if len(multitypes) > 1:
        logger.warning(
            f"Multiple types found for event {event.display_name}. "
            + f"Will use the first source of type {dtype} and ignore others: {','.join(multitypes)}"
        )

    return dtype


# other


def _sv_loader(config: ConfigProvider, sep) -> pd.DataFrame:
    """General loader for CSV or TSV files"""
    import polars as pl

    try:
        events = pl.read_csv(config.event_path, separator=sep).rename(
            {config.ev_type: "Type", config.ev_time: "Time", config.ev_value: "Value"}
        )

        # since importing CSVs automatically cast numbers to ints, make sure the columns
        # shared with predictions become strings so we don't have a type mismatch
        defined_types = config.prediction_types
        usage = config.usage
        cast_cols = []
        for col in [usage.entity_id, usage.context_id, usage.predict_time]:
            if col is not None and col in events.columns and defined_types.get(col) == "object":
                cast_cols.append(pl.col(col).cast(pl.String))
        if cast_cols:
            events = events.with_columns(cast_cols)
    except BaseException:
        logger.warning(f"No events found at {config.event_path}")
        events = pl.DataFrame({col: [] for col in config.entity_keys + ["Type", "Time", "Value"]})

    return events
