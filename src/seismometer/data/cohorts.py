from typing import List, Optional, Union

import numpy as np
import polars as pl

from .decorators import export
from .performance import calculate_bin_stats

SeriesOrArray = Union[pl.Series, np.ndarray]

# region Stats


@export
def get_cohort_data(
    df: pl.DataFrame,
    cohort_feature: str,
    *,
    proba: Union[str, SeriesOrArray],
    true: Union[str, SeriesOrArray] = "TARGET",
    splits: Optional[List] = None,
) -> pl.DataFrame:
    """
    Convenience function to create and format data for use in the cohort plots.
    Takes in information about the class, predictions, and true labels to output a dataset and corresponding labels.

    In the case that multiple columns are used, predictions from each column are appended to the result
    so that rows sharing a cohort group are disjoint, and rows with different cohort columns potentially overlap.

    Currently supports cohort_features of type Categorical (splits all categories) and Numeric (splits on specified
    values or at mean).

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe of observations to use for plotting, must contain the column specified in cohort_feature.
        Additionally, must contain columns specified by proba and true if using strings and not arrays.
    cohort_feature : str
        string specification of the dataframe column to split.
        Currently supports numeric and categorical columns.
    proba : Union[str, SeriesOrArray]
        The predictions made by the model under review.

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe.

    true : Union[str, SeriesOrArray]
        The true label associated with a prediction, by default "TARGET".

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe and int values.

    splits : Optional[List]
        The numeric values to split cohorts or category values to include, treats each category value as its own
        split, by default None.
        If None, will create a dichotomy for numeric values split at the mean.

    Returns
    -------
    pl.DataFrame
        Data - ingestible by plot_cohort_* functions.
    """
    # Auto-convert pandas DataFrame to Polars for backward compatibility
    if not isinstance(df, pl.DataFrame):
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

    # Find data
    proba_series = resolve_col_data(df, proba).cast(pl.Float64)
    true_series = resolve_col_data(df, true).cast(pl.Float64)  # handle nan but require numeric

    cohort_series = resolve_cohorts(df[cohort_feature], splits)

    # Create standard DataFrame
    rv = pl.DataFrame({"true": true_series, "pred": proba_series, "cohort": cohort_series})

    return rv.drop_nulls()


@export
def get_cohort_performance_data(
    df: pl.DataFrame,
    cohort_feature: str,
    *,
    proba: Union[str, SeriesOrArray],
    true: Union[str, SeriesOrArray] = "TARGET",
    splits: Optional[List] = None,
    censor_threshold: int = 10,
) -> pl.DataFrame:
    """
    Generates a dataframe with particular performance metrics (accuracy, sensitivity,
    specificity, ppv, npv, and flag rate (predicted positive condition rate)) for
    particular threshold values and cohort.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe of observations to use, must contain the column specified in cohort_feature.
        Additionally, must contain columns specified by proba and true if using strings and not arrays.
    cohort_feature : str
        String specification of the dataframe column to split.
        Currently supports numeric and categorical columns.
    proba : Union[str, SeriesOrArray]
        The predictions made by the model under review.

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe.

    true : Union[str, SeriesOrArray], default="TARGET"
        The true label being predicted.

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe and int values.

    splits : Optional[List], default=None
        Optional - the numeric values to split cohorts or category values to include, treats each category value as its
        own split.
        If None, will create a dichotomy for numeric values split at the mean.
    censor_threshold : int, default=10
        Minimum number of observations in a cohort to calculate performance metrics.

    Returns
    -------
    pl.DataFrame
        Performance statistics for particular threshold values by cohort.
    """
    import pandas as pd

    data = get_cohort_data(df, cohort_feature, proba=proba, true=true, splits=splits)

    cohort_perf_stats = []
    observed = set()
    data = data.with_columns(pl.col("true").cast(pl.Int64))

    # Get all unique cohort labels
    all_cohorts = data["cohort"].unique().drop_nulls().to_list()

    for cohort_df in data.partition_by("cohort", as_dict=False):
        if cohort_df.height == 0:
            continue

        label = cohort_df["cohort"][0]
        true_sum = cohort_df["true"].sum()
        true_count = cohort_df.height

        if true_sum == 0 or true_count < censor_threshold:
            continue

        # calculate_bin_stats expects pandas Series - convert only the needed data
        true_series = pd.Series(cohort_df["true"].to_numpy())
        pred_series = pd.Series(cohort_df["pred"].to_numpy())

        ind_perf_stats = calculate_bin_stats(true_series, pred_series)
        ind_perf_stats["cohort"] = label
        ind_perf_stats["cohort-count"] = true_count
        ind_perf_stats["cohort-targetcount"] = true_sum

        cohort_perf_stats.append(ind_perf_stats)
        observed.add(label)

    # Add empty cohorts
    for label in set(all_cohorts) - observed:
        cohort_perf_stats.append(
            pd.DataFrame({"cohort": label, "cohort-count": 0, "cohort-targetcount": 0}, index=[0])
        )

    if not cohort_perf_stats:
        return pl.DataFrame()

    frame = pd.concat(cohort_perf_stats, ignore_index=True)
    # Convert back to Polars
    return pl.from_pandas(frame)


def resolve_col_data(df: pl.DataFrame, feature: Union[str, pl.Series]) -> pl.Series:
    """
    Handles resolving feature from either being a series or specifying a series in the dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        Containing a column of name feature if feature is passed in as a string.
    feature : Union[str, pl.Series]
        Either a polars.Series or a column name in the dataframe.

    Returns
    -------
    pl.Series.
    """

    if isinstance(feature, str):
        if feature in df.columns:
            col = df[feature]
            # Polars uses .clone(), pandas uses .copy()
            return col.clone() if hasattr(col, "clone") else col.copy()
        else:
            raise KeyError(f"Feature {feature} was not found in dataframe")
    elif hasattr(feature, "ndim"):
        if feature.ndim > 1:  # probas from sklearn is nx2 with second column being the positive predictions
            return pl.Series(feature[:, 1])
        else:
            return pl.Series(feature) if not isinstance(feature, pl.Series) else feature
    else:
        raise TypeError("Feature must be a string or polars.Series, was given a ", type(feature))


# endregion
# region Labels


@export
def resolve_cohorts(series: SeriesOrArray, splits: Optional[List] = None) -> pl.Series:
    """
    Bin a series of data based on the defined splits if defined.
    Only handles numeric and categorical data.

    Parameters
    ----------
    series : SeriesOrArray
        polars series of data to bin.
    splits : Optional[List], optional
        List of splits to define inner bins (default: None).

    Returns
    -------
    pl.Series
        Categorical series with labels.
    """
    # Check if series is categorical (Polars Categorical dtype)
    if isinstance(series.dtype, pl.Categorical) or series.dtype == pl.Categorical:
        return label_cohorts_categorical(series, splits)
    # Treat everything else like continuous - can raise errors with unexpected data types
    return label_cohorts_numeric(series, splits)


def label_cohorts_numeric(series: SeriesOrArray, splits: Optional[List] = None) -> pl.Series:
    """
    Bin a continuous numeric series of data, based on thresholds of inner bin edges.

    Parameters
    ----------
    series : SeriesOrArray
        polars series of data to bin.
    splits : Optional[List], optional
        List of splits to define inner bins (default: None-> series.mean()).

    Returns:
    --------
    pl.Series
        Categorical series with labels.
    """
    # Convert to numpy for binning
    series_np = series.to_numpy()
    bins = find_bin_edges(series_np, splits)
    bin_ixs = np.digitize(series_np, bins, right=False)
    has_good_binning(bin_ixs, bins)

    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)] + [f">={bins[-1]}"]
    labels[0] = f"<{bins[1]}"

    # Map bin indices to labels
    label_values = [labels[ix - 1] for ix in bin_ixs]
    return pl.Series(label_values, dtype=pl.Categorical)


def has_good_binning(bin_ixs: List, bin_edges: List) -> None:
    """
    Verifies that the binning is sound by making sure lists are equal length.

       - If there are fewer realized ix than edges then a bin is empty.
       - If there are more realized ix than edges then the edge array got out of sync somehow.

    Parameters
    ----------
    bin_ixs : List
        List of ix for binned values; output of np.digitize using bin_edges.
    bin_edges : List
        List of bin edges to split on.

    Raises
    ------
    IndexError
        The list of indexes does not align with the bin edge list.
    """
    if len(bin_edges) != len(np.unique(bin_ixs)):
        raise IndexError("Splits provided contain some empty bins.")


def label_cohorts_categorical(series: SeriesOrArray, cat_values: Optional[list] = None) -> pl.Series:
    """
    Bin a categorical series of data, reduced to a set of category values.

    Parameters
    ----------
    series : SeriesOrArray
        polars series of data to bin.
    cat_values : Optional[list], optional
        List of categories to reduce to (default: None-> all observed categories).

    Returns
    -------
    pl.Series
        Categorical series with filtered categories.
    """
    # Ensure we have a Series, not an Expr
    if isinstance(series, pl.Expr):
        # If it's an Expr, we need to evaluate it first - this shouldn't happen in normal usage
        raise TypeError(f"Expected pl.Series, got {type(series)}")

    # If no splits specified, return all observed values as categorical
    if cat_values is None:
        return series.cast(pl.Categorical)

    # Get unique categories in the series
    unique_cats = series.unique().drop_nulls().to_list()

    # If the series has exactly the requested categories, return it
    if set(cat_values) == set(unique_cats):
        return series.cast(pl.Categorical)

    # Filter to only the requested categories
    if cat_values is not None:
        # Create new series with values not in cat_values set to None
        df_temp = pl.DataFrame({"col": series})
        filtered = df_temp.select(pl.when(pl.col("col").is_in(cat_values)).then(pl.col("col")).otherwise(None))["col"]
        return filtered.cast(pl.Categorical)


def find_bin_edges(series: SeriesOrArray, thresholds: Optional[list] = None) -> list[float]:
    """
    Creates list of bin edges from a series of continuous numeric data and list of inner thresholds.
    Contains lower bound but does not need upper bound due to numpy handling already understanding greater than max.

    Parameters
    ----------
    series : SeriesOrArray
        pandas series of data to bin.
    thresholds : Optional[list], optional
        List of thresholds indicating inner bin edges (default: None-> series.mean()).

    Returns
    -------
    list[float]
        Sorted list of bin edges.
    """
    if not thresholds:
        thresholds = [np.mean(series)]
    # Ensure list-like; handle float
    if not hasattr(thresholds, "insert"):
        thresholds = [thresholds]
    # Ensure sorted and unique entries
    bins = sorted(set(thresholds))

    ymin = min(series)
    if bins[0] > ymin:
        bins.insert(0, ymin)

    return bins


# endregion
