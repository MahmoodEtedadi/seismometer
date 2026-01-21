import polars as pl

from seismometer.data import pandas_helpers as pdh

from .decorators import export


@export
def default_cohort_summaries(
    dataframe: pl.DataFrame, attribute: str, options: list[str], entity_id_col: str
) -> pl.DataFrame:
    """
    Generate a dataframe of summary counts from the input dataframe.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input dataframe.
    attribute : str
        The attribute to generate summary levels for.
    options : list[str]
        An ordered list of options to reindex the dataframe on.
    entity_id_col : str
        The column name for the dataframe column containing the entity identifier.

    Returns
    -------
    pl.DataFrame
        A dataframe of summary counts.
    """
    from seismometer.seismogram import Seismogram

    sg = Seismogram()

    # Count predictions by attribute
    left = dataframe.group_by(attribute, maintain_order=True).agg(pl.len().alias("Predictions"))

    # Get event scores and count entities by attribute
    event_df = pdh.event_score(
        dataframe, [entity_id_col], sg.output, sg.predict_time, sg.target, sg.event_aggregation_method(sg.target)
    )
    right = event_df.group_by(attribute, maintain_order=True).agg(pl.len().alias("Entities"))

    # Join the two counts and reindex to match options
    result = left.join(right, on=attribute, how="full", coalesce=True)

    # Create a dataframe with all options to ensure we have all rows
    # Match the type of the result's attribute column
    options_df = pl.DataFrame({attribute: options})
    result_col_type = result[attribute].dtype
    if result_col_type != options_df[attribute].dtype:
        options_df = options_df.with_columns(pl.col(attribute).cast(result_col_type))
    result = options_df.join(result, on=attribute, how="left", coalesce=True)

    # Cast count columns to Int64 for consistency
    result = result.with_columns(
        [pl.col("Predictions").cast(pl.Int64).fill_null(0), pl.col("Entities").cast(pl.Int64).fill_null(0)]
    )

    return result


@export
def score_target_cohort_summaries(
    dataframe: pl.DataFrame,
    groupby_groups: list[str],
    grab_groups: list[str],
    entity_id_col: str,
) -> pl.DataFrame:
    """
    Generate a dataframe of summary counts from the input dataframe.
    Also, summarizes by additional summary levels in groupby_groups.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input dataframe.
    groupby_groups : list[str]
        Selections to groupby when generating summaries (attribute, score bins, target, etc.).
    grab_groups : list[str]
        Columns to grab while summarizing.
    entity_id_col : str
        The column name for the dataframe column containing the entity identifier.

    Returns
    -------
    pl.DataFrame
        A dataframe of summary counts.
    """
    import pandas as pd

    from seismometer.seismogram import Seismogram

    sg = Seismogram()

    # Handle pandas Series in groupby_groups (e.g., from pd.cut)
    # Convert them to Polars columns and add them to the dataframe
    temp_cols = {}
    group_col_names = []
    temp_to_grab_mapping = {}  # Map temp column names to their corresponding grab_groups names
    temp_series_data = {}  # Store original pandas Series for later reconstruction

    # Build a list of grab_groups columns that are not regular columns in groupby_groups
    # These are the columns that pandas Series should map to
    # Filter out pandas Series from groupby_groups for the comparison
    groupby_column_names = [g for g in groupby_groups if isinstance(g, str)]
    grab_groups_not_in_groupby = [g for g in grab_groups if g not in groupby_column_names]
    grab_idx = 0

    for i, group in enumerate(groupby_groups):
        if isinstance(group, pd.Series):
            # Generate a temporary column name
            temp_col_name = f"_temp_group_{i}"
            # Store the original series for later
            temp_series_data[temp_col_name] = group
            # Convert to list to handle categorical/interval types
            temp_cols[temp_col_name] = pl.Series(temp_col_name, group.astype(str).tolist())
            group_col_names.append(temp_col_name)
            # Map temp column to the next available grab_groups column that's not in groupby_groups
            if grab_idx < len(grab_groups_not_in_groupby):
                temp_to_grab_mapping[temp_col_name] = grab_groups_not_in_groupby[grab_idx]
                grab_idx += 1
        else:
            group_col_names.append(group)

    # Add temporary columns to dataframe if any
    df_with_temp = dataframe
    if temp_cols:
        df_with_temp = dataframe.with_columns(list(temp_cols.values()))

    # Count predictions by groups
    predictions = (
        df_with_temp.select(grab_groups + list(temp_cols.keys()))
        .group_by(group_col_names, maintain_order=True)
        .agg(pl.len().alias("Predictions"))
    )

    # Get event scores and count entities
    # For this function, we need to ensure we get a Polars DataFrame back
    # Since we're passing a Polars dataframe and need to do Polars operations on the result
    df = pdh.event_score(
        dataframe, [entity_id_col], sg.output, sg.predict_time, sg.target, sg.event_aggregation_method(sg.target)
    )
    # Ensure df is Polars (in case event_score returned pandas for test compatibility)
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # Add temporary columns to event score df by re-applying the transformations
    df_with_temp2 = df
    if temp_cols:
        # For each temp column, we need to re-create it based on the values in the event_score result
        # Create a mapping from the original data values to their bin labels
        for temp_col_name, original_series in temp_series_data.items():
            # Find which grab_group this temp column corresponds to
            grab_col = temp_to_grab_mapping.get(temp_col_name)
            if grab_col and grab_col in df_with_temp2.columns:
                # Create a mapping DataFrame from values in the original column to their binned versions
                # Convert pandas Series to list if needed
                if hasattr(original_series, "values"):
                    bin_values = original_series.values
                else:
                    bin_values = list(original_series)

                mapping_df = pl.DataFrame(
                    {grab_col: dataframe[grab_col].to_list(), temp_col_name: [str(v) for v in bin_values]}
                ).unique(subset=[grab_col])

                # Only add if the temp column doesn't already exist
                if temp_col_name not in df_with_temp2.columns:
                    # Join to add the mapped column
                    df_with_temp2 = df_with_temp2.join(mapping_df, on=grab_col, how="left")

    entities = (
        df_with_temp2.select(grab_groups + list(temp_cols.keys()))
        .group_by(group_col_names, maintain_order=True)
        .agg(pl.len().alias("Entities"))
        .cast({"Entities": pl.Int64})
    )

    # Join predictions and entities
    result = predictions.join(entities, on=group_col_names, how="full", coalesce=True).fill_null(0)

    # Rename temporary columns back to their original grab_groups names
    # Build the sort columns list based on renamed columns
    sort_cols = group_col_names.copy()
    if temp_to_grab_mapping:
        result = result.rename(temp_to_grab_mapping)
        # Update sort_cols to use the renamed columns
        sort_cols = [temp_to_grab_mapping.get(col, col) for col in group_col_names]

    # Sort by the group columns to ensure consistent ordering
    result = result.sort(sort_cols)

    return result
