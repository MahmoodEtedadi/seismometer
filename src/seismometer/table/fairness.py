import logging
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import polars as pl
import traitlets
from great_tables import GT, html, loc, style
from ipywidgets import HTML, Box, FloatSlider, Layout, ValueWidget, VBox

from seismometer.controls.explore import ExplorationWidget, ModelOptionsWidget
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title
from seismometer.core.autometrics import store_call_parameters
from seismometer.data import metric_apis
from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.data.performance import BinaryClassifierMetricGenerator, MetricGenerator

logger = logging.getLogger("seismometer")

COUNT = "Count"
COHORT = "Cohort"
CLASS = "Class"


# region Fairness Icons
class FairnessIcons(Enum):
    """
    Enum for fairness icons
    """

    DEFAULT = "🔹"
    GOOD = "🟢"
    UNKNOWN = "❔"
    WARNING_HIGH = "🔼"
    WARNING_LOW = "🔽"
    CRITICAL_HIGH = "🔺"
    CRITICAL_LOW = "🔻"

    @classmethod
    def get_fairness_legend(cls, limit: float = 0.25, *, open: bool = True, censor_threshold: int = 10) -> str:
        return html(
            f"""
<details {'open' if open else ''}><summary><span style="font-size: 100%; font-weight: bold;">Legend</span></summary>
<table>
<tr style="background: none;">
<td style="text-align: left;">{cls.DEFAULT.value} The default cohort for the category.</td>
<td style="text-align: left;">{cls.GOOD.value} Within {limit:.2%} of the default cohort.</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.WARNING_LOW.value} Within {2*limit:.2%} lower than the default cohort.</td>
<td style="text-align: left;">{cls.WARNING_HIGH.value} Within {2*limit:.2%} greater than the default cohort.</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.CRITICAL_LOW.value} More than {2*limit:.2%} lower than the default cohort.</td>
<td style="text-align: left;">{cls.CRITICAL_HIGH.value} More than {2*limit:.2%} greater than the default cohort.</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.UNKNOWN.value} Censored, fewer than {censor_threshold} observations.</td>
</tr>
</details>"""
        )

    @classmethod
    def get_fairness_icon(cls, ratio, limit: float = 0.25) -> "FairnessIcons":
        """
        Icon for fairness ratio
        If fairness ratio is 0.25 (25%) we want to show a warning if we are outside this range and
        a critical warning if we are 2x outside this range

        We are looking at 1 / (1 + limit) < ratio < 1 + limit

        For a limit of 0.25 we are looking at 0.80 < ratio < 1.25 (25% bigger, or 20% smaller)
        Alternatively for a limit of 0.50 we are looking at 0.67 < ratio < 1.50 (50% bigger, or 33% smaller)
        For a limit of 1.0 we are looking at 0.5 < ratio < 2.0 (100% bigger, or 50% smaller) which allows a 2x
        difference between a one group and another.

        The extended upper bounds are at 1 + limit and 1 + 2*limit.

        Parameters
        ----------
        ratio : float
            Ratio of the cohort to the largest cohort
        limit : float, optional
            Allowed percentage difference by cohort, by default 0.25, measured from the smaller metric to the larger.

        Returns
        -------
        FairnessIcons
            Icon for the ratio based on the limit.
        """
        upper_limit, twice_upper_limit = 1 + limit, 1 + 2 * limit
        lower_limit, twice_lower_limit = 1 / upper_limit, 1 / twice_upper_limit

        if ratio is None or np.isnan(ratio):
            return FairnessIcons.UNKNOWN
        if ratio > twice_upper_limit:
            return FairnessIcons.CRITICAL_HIGH
        if ratio < twice_lower_limit:
            return FairnessIcons.CRITICAL_LOW
        if ratio > upper_limit:
            return FairnessIcons.WARNING_HIGH
        if ratio < lower_limit:
            return FairnessIcons.WARNING_LOW
        if ratio == 1:
            return FairnessIcons.DEFAULT
        return FairnessIcons.GOOD


# endregion
# region Fairness Table


def sort_fairness_table(dataframe: pl.DataFrame, cohort_groups: tuple[str]):
    """
    Generates a sort key for the fairness table based on Cohort group name and Count

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame to sort
    cohort_groups : tuple[str]
        Cohort group names for sorting.
    """
    # Create sort keys for cohort and count
    cohort_order_map = {cohort: idx for idx, cohort in enumerate(cohort_groups)}

    # Use map_elements to handle mixed types in Count column
    def count_sort_key(val):
        if val in [FairnessIcons.UNKNOWN.value, "--", None]:
            return 0
        try:
            return -int(val)
        except (ValueError, TypeError):
            return 0

    dataframe = dataframe.with_columns(
        [
            pl.col(COHORT)
            .map_elements(lambda x: cohort_order_map.get(x, 999), return_dtype=pl.Int32)
            .alias("_cohort_order"),
            pl.col(COUNT).map_elements(count_sort_key, return_dtype=pl.Int64).alias("_count_order"),
        ]
    )

    result = dataframe.sort(["_cohort_order", "_count_order"]).drop(["_cohort_order", "_count_order"])
    return result


def fairness_table(
    dataframe: pl.DataFrame,
    metric_fn: Callable[..., dict[str, float]],
    metric_list: list[str] = None,
    fairness_ratio: float = 0.25,
    cohort_dict: dict[str, tuple[Any]] = None,
    *,
    censor_threshold: int = 10,
    rho: float = 0.5,
    **kwargs,
) -> HTML:
    """
    Fairness table for evaluating metrics across cohorts, found by taking the largest subgroup within each cohort
    as the default cohort and taking the ratio between a subgroup's metric and the default group's metric.

    For example if if a cohort has three classes A, B, and C with counts of 10, 20, and 30 respectively, the default
    cohort would be C. For a metric M, we would calculate M(A), M(B) and M(C) and then calculate the ratios
    M(A)/M(C) and its reciprocal M(C)/M(A).

    If M(A)/M(C) > 1 + limit, then cohort A will be flagged as higher than the default.
    If M(A)/M(C) > 1 + 2 * limit, then cohort A will be flagged as critically higher than the default.

    If M(C)/M(A) > 1 + limit, then cohort A will be flagged as lower than the default.
    If M(C)/M(A) > 1 + 2 * limit, then cohort A will be flagged as critically lower than the default.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Source data to generate a fairness table for
    metric_fn : Callable[..., dict[str, float]]
        Metric function to generate raw metrics, which MUST be positive values.
    metric_list : list[str]
        List of metrics to use from the metric function
    fairness_ratio : float
        Ratio of acceptable difference between cohorts, 20% is 0.2, 200% is 2.0.
        Bound is multiplicatively symmetric around 1, so 200% means up to 3x larger or 3x smaller (1/3 the original).
        A typical bound is 0.25 (1.25x larger or 0.8x smaller)
    cohort_dict : dict[str, tuple[Any]]
        collection of cohort groups to loop over
    censor_threshold : int, optional
        Limit at which a cohort group will be removed from the table if not enough observations are found,
        by default 10.

    Returns
    -------
    HTML
        The HTML table for the fairness evaluation
    """
    fairness_groups = []
    metric_groups = []
    icon_groups = []

    if fairness_ratio <= 0:
        raise ValueError("Fairness ratio must be greater than 0")

    if not cohort_dict:
        raise ValueError("No cohorts provided for fairness evaluation")

    recorder = metric_apis.OpenTelemetryRecorder(metric_names=metric_list, name="Fairness Table Metric Generator")

    for cohort_column in cohort_dict:
        cohort_rows = []
        for cohort_class in cohort_dict[cohort_column]:
            cohort_filter = FilterRule.eq(cohort_column, cohort_class)
            cohort_dataframe = cohort_filter.filter(dataframe)

            row_data = {COHORT: cohort_column, CLASS: cohort_class, COUNT: len(cohort_dataframe)}

            # Add all of the information we can reasonably find.
            attribute_info = {"fairness_ratio": fairness_ratio}
            for attr in "score_threshold", "target_col", "score_col":
                if attr in kwargs:
                    attribute_info |= {attr: kwargs[attr]}
            rho_info = {"rho": rho}
            metrics = metric_fn(cohort_dataframe, metric_list, **kwargs)
            recorder.populate_metrics({cohort_column: cohort_class} | attribute_info | rho_info, metrics)
            row_data.update(metrics)

            cohort_rows.append(row_data)

        # Create Polars DataFrame
        cohort_data = pl.DataFrame(cohort_rows)

        # Find the row with maximum count
        max_count_row = cohort_data.filter(pl.col(COUNT) == pl.col(COUNT).max()).row(0, named=True)

        # Calculate ratios by dividing each metric by the max count row's metric
        ratio_exprs = [pl.col(COUNT).alias(COUNT)]
        for metric in metric_list:
            max_val = max_count_row[metric]
            ratio_exprs.append((pl.col(metric) / max_val).alias(metric))
        cohort_ratios = cohort_data.select([COHORT, CLASS] + ratio_exprs)

        # Generate fairness icons - store as strings directly
        icon_data = cohort_ratios.clone()
        for metric in metric_list:
            icon_data = icon_data.with_columns(
                pl.col(metric)
                .map_elements(
                    lambda x: FairnessIcons.get_fairness_icon(x, fairness_ratio).value, return_dtype=pl.String
                )
                .alias(metric)
            )

        fairness_groups.append(cohort_ratios)
        icon_groups.append(icon_data)
        metric_groups.append(cohort_data)

    # Concatenate all cohort groups
    fairness_data = pl.concat(fairness_groups)
    metric_data = pl.concat(metric_groups)
    fairness_icons = pl.concat(icon_groups)

    # Apply censoring for small cohorts
    # Create a boolean column to mark censored rows (must do this while COUNT is still numeric)
    metric_data = metric_data.with_columns((pl.col(COUNT) < censor_threshold).alias("_censored"))
    fairness_data = fairness_data.with_columns((pl.col(COUNT) < censor_threshold).alias("_censored"))
    fairness_icons = fairness_icons.with_columns((pl.col(COUNT) < censor_threshold).alias("_censored"))

    # Update metric_data: set metrics to null when censored
    for metric in metric_list:
        metric_data = metric_data.with_columns(
            pl.when(pl.col("_censored")).then(None).otherwise(pl.col(metric)).alias(metric)
        )

    # Update fairness_data: set metrics to null when censored
    for metric in metric_list:
        fairness_data = fairness_data.with_columns(
            pl.when(pl.col("_censored")).then(None).otherwise(pl.col(metric)).alias(metric)
        )

    # Update fairness_icons: set COUNT and metrics to UNKNOWN when censored
    fairness_icons = fairness_icons.with_columns(
        pl.when(pl.col("_censored"))
        .then(pl.lit(FairnessIcons.UNKNOWN.value))
        .otherwise(pl.col(COUNT).cast(pl.String))
        .alias(COUNT)
    )
    for metric in metric_list:
        fairness_icons = fairness_icons.with_columns(
            pl.when(pl.col("_censored"))
            .then(pl.lit(FairnessIcons.UNKNOWN.value))
            .otherwise(pl.col(metric))
            .alias(metric)
        )

    # Drop the temporary censored column
    metric_data = metric_data.drop("_censored")
    fairness_data = fairness_data.drop("_censored")
    fairness_icons = fairness_icons.drop("_censored")

    # Format metric values as strings
    metric_data_formatted = metric_data.select(
        [COHORT, CLASS]
        + [
            pl.when(pl.col(metric).is_null())
            .then(pl.lit(""))
            .otherwise(pl.col(metric).map_elements(lambda x: f"  {x:.2f}  ", return_dtype=pl.String))
            .alias(f"{metric}_metric")
            for metric in metric_list
        ]
    )

    # Format fairness ratios as strings
    fairness_data_formatted = fairness_data.select(
        [COHORT, CLASS]
        + [
            pl.when(pl.col(metric).is_null() | (pl.col(metric) == 1.0))
            .then(pl.lit(""))
            .otherwise((pl.col(metric) - 1).map_elements(lambda x: f"  ({x:.2%})  ", return_dtype=pl.String))
            .alias(f"{metric}_fairness")
            for metric in metric_list
        ]
    )

    # Convert icon strings: replace UNKNOWN emoji with "--" for display
    for metric in metric_list:
        fairness_icons = fairness_icons.with_columns(
            pl.when(pl.col(metric) == FairnessIcons.UNKNOWN.value)
            .then(pl.lit("--"))
            .otherwise(pl.col(metric))
            .alias(metric)
        )

    # Join the formatted metric and fairness data
    fairness_icons = fairness_icons.join(metric_data_formatted, on=[COHORT, CLASS], how="left")
    fairness_icons = fairness_icons.join(fairness_data_formatted, on=[COHORT, CLASS], how="left")

    # Concatenate strings for each metric
    for metric in metric_list:
        fairness_icons = fairness_icons.with_columns(
            (
                pl.col(metric) + pl.col(f"{metric}_metric").fill_null("") + pl.col(f"{metric}_fairness").fill_null("")
            ).alias(metric)
        ).drop([f"{metric}_metric", f"{metric}_fairness"])

    legend = FairnessIcons.get_fairness_legend(fairness_ratio, censor_threshold=censor_threshold)

    table_data = fairness_icons.select([COHORT, CLASS, COUNT] + metric_list)
    table_data = sort_fairness_table(table_data, list(cohort_dict.keys()))

    table_html = (
        GT(table_data)
        .tab_stub(groupname_col=COHORT, rowname_col=CLASS)
        .tab_style(
            style=style.text(align="center"),
            locations=loc.column_header(),
        )
        .tab_style(
            style=style.borders(sides=["right"], weight="1px", color="#D3D3D3"),
            locations=loc.body(columns=[COUNT] + metric_list),
        )
        .tab_source_note(source_note=legend)
        .opt_horizontal_padding(scale=3)
        .tab_options(row_group_font_weight="bold")
        .cols_align(align="left")
        .cols_align(align="right", columns=[COUNT])
    ).as_raw_html()
    return HTML(table_html, layout=Layout(max_height="800px"))


def _autometric_plot_binary_classifier_metrics(
    metric_generator: float,
    metric_list: list[str],
    cohort_dict: dict[str, tuple[Any]],
    fairness_ratio: float,
    target: str,
    score: str,
    threshold: float,
    *,
    per_context=False,
):
    """Serves only as a wrapper of plot_binary_classifier_metrics so that
    we don't have to serialize a metric generator object.

    Parameters
    ----------
    metric_generator: float between 0 and 1
        Probability of a treatment being effective. This is named metric_generator
        instead of rho because it is an internal method and having the object be
        the same name as what it is replacing in the real method makes
        serialization much easier.
    """
    bcmg = BinaryClassifierMetricGenerator(rho=metric_generator)
    binary_metrics_fairness_table(
        bcmg, metric_list, cohort_dict, fairness_ratio, target, score, threshold, per_context=per_context
    )


# endregion
# region Fairness Table Wrapper
@store_call_parameters(cohort_dict="cohort_dict")
def binary_metrics_fairness_table(
    metric_generator: BinaryClassifierMetricGenerator,
    metric_list: list[str],
    cohort_dict: dict[str, tuple[Any]],
    fairness_ratio: float,
    target: str,
    score: str,
    threshold: float,
    *,
    per_context=False,
) -> HTML:
    """
    Binary fairness metrics table

    Parameters
    ----------
    metric_generator : The BinaryClassifierMetricGenerator that determines rho.
    metric_list : list[str]
        List of metrics to evaluate.
    cohort_dict : dict[str, tuple[Any]]
        Collection of cohort groups to loop over.
    fairness_ratio : float
        Ratio of acceptable difference between cohorts, 20% is 0.2, 200% is 2.0.
    target : str
        The target descriptor for the binary classifier.
    score : str
        The score descriptor for the binary classifier.
    threshold : float
        The threshold for the binary classifier.
    per_context : bool, optional
        Whether to group scores by context, by default False.

    Returns
    -------
    HTML
        The HTML table for the fairness evaluation.
    """
    from seismometer.seismogram import Seismogram

    sg = Seismogram()
    target_column = pdh.event_value(target)
    data = (
        pdh.event_score(
            sg.dataframe,
            sg.entity_keys,
            score=score,
            ref_time=sg.predict_time,
            ref_event=target,
            aggregation_method=sg.event_aggregation_method(target),
        )
        if per_context
        else sg.dataframe
    )
    return fairness_table(
        data,
        metric_generator,
        metric_list,
        fairness_ratio,
        cohort_dict,
        censor_threshold=sg.censor_threshold,
        target_col=target_column,
        score_col=score,
        score_threshold=threshold,
        rho=metric_generator.rho,
    )


def custom_metrics_fairness_table(metric_generator, metric_list, cohort_dict, fairness_ratio) -> HTML:
    """
    For use by fairness tables that need custom metric generators.

    Parameters
    ----------
    metric_generator : MetricGenerator
        Metric generator to use for the fairness table.
    metric_list : list[str]
        List of metrics to evaluate.
    cohort_dict : dict[str, tuple[Any]]
        Collection of cohort groups to loop over.
    fairness_ratio : float
        Ratio of acceptable difference between cohorts, 20% is 0.2, 200% is 2.0.

    Returns
    -------
    HTML
        The HTML table for the fairness evaluation.
    """
    from seismometer.seismogram import Seismogram

    sg = Seismogram()
    dataframe = sg.dataframe
    if not cohort_dict:
        cohort_dict = sg.available_cohort_groups
    return fairness_table(
        dataframe, metric_generator, metric_list, fairness_ratio, cohort_dict, censor_threshold=sg.censor_threshold
    )


# endregion

# region Fairness Controls


class FairnessOptionsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the model options.")

    def __init__(
        self,
        metric_names: tuple[str],
        cohort_dict: dict[str, tuple[Any]],
        fairness_ratio: float = 0.25,
        *,
        model_options_widget=None,
        default_metrics=None,
    ):
        """
        Widget for selecting fairness options

        Parameters
        ----------
        metric_names : tuple[str]
            Metrics that can be evaluated for fairness.
        cohort_dict : dict[str, tuple[Any]]
            Dictionary of cohort groups.
        fairness_ratio : float, optional
            Allowed difference by cohort, by default 0.25
        model_options_widget : Optional[widget], optional
            Additional model options if needed, will appear before fairness options, by default None.
        default_metrics : Optional[tuple[str]], optional
            Default list of metrics to select initially for fairness evaluation, by default None.
        """
        self.model_options_widget = model_options_widget
        default_metrics = default_metrics or metric_names
        self.metric_list = MultiselectDropdownWidget(metric_names, value=default_metrics, title="Fairness Metrics")
        self.cohort_list = MultiSelectionListWidget(
            cohort_dict,
            title="Cohorts",
        )
        self.fairness_slider = FloatSlider(
            min=0.01,
            max=1.00,
            step=0.01,
            value=fairness_ratio,
            description="Threshold",
            style=WIDE_LABEL_STYLE,
        )
        self.all_cohorts = cohort_dict
        self.metric_list.observe(self._on_value_changed, names="value")
        self.cohort_list.observe(self._on_value_changed, names="value")
        self.fairness_slider.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Fairness Options"),
            self.fairness_slider,
            self.metric_list,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        super().__init__(
            children=[
                VBox(children=v_children, layout=Layout(align_items="flex-end", flex="0 0 auto")),
                self.cohort_list,
            ],
            layout=BOX_GRID_LAYOUT,
        )

        self._on_value_changed()
        self._disabled = False

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        self.metric_list.disabled = value
        self.cohort_list.disabled = value
        self.fairness_slider.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        new_value = {
            "metric_list": self.metric_list.value,
            "cohort_list": self.cohort_list.value,
            "fairness_ratio": self.fairness_slider.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def metrics(self):
        return self.metric_list.value

    @property
    def cohorts(self):
        return self.cohort_list.value or self.all_cohorts

    @property
    def fairness_ratio(self):
        return self.fairness_slider.value

    @property
    def model_options(self):
        return self.model_options_widget if self.model_options_widget else None


class ExplorationFairnessWidget(ExplorationWidget):
    """
    A widget for exploring model fairness across cohorts
    """

    def __init__(self, metrics: MetricGenerator):
        """
        Exploration widget for model fairness evaluation based on cohort selection.
        Only works for global model metrics, not metrics that rely on parameters.

        Parameters
        ----------
        metrics : list[MetricGenerator] or MetricGenerator
            list of metric functions to evaluate for fairness
        """

        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.metrics_generator = metrics
        metric_names = [name for name in metrics.metric_names]

        super().__init__(
            title="Fairness Audit",
            option_widget=FairnessOptionsWidget(
                metric_names,
                sg.available_cohort_groups,
                fairness_ratio=0.25,
            ),
            plot_function=custom_metrics_fairness_table,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot"""
        args = (
            self.metrics_generator,
            list(self.option_widget.metrics),
            self.option_widget.cohorts,
            self.option_widget.fairness_ratio,
        )
        return args, {}


class ExploreBinaryModelFairness(ExplorationWidget):
    """
    A widget for exploring model fairness across cohorts for a binary classifier
    """

    def __init__(self, rho: Optional[float] = None):
        """
        Exploration widget for model evaluation, showing a plot for a given target,
        score, threshold, and cohort selection.

        Parameters
        ----------
        rho : Optional[float], between 0 and 1
            treatment efficacy as a probability of positive result.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.metric_generator = BinaryClassifierMetricGenerator(rho=rho)
        metric_names = tuple(self.metric_generator.metric_names)
        model_options_widget = ModelOptionsWidget(
            sg.target_cols, sg.output_list, {"Score Threshold": max(sg.thresholds)}, per_context=False
        )

        super().__init__(
            title="Binary Classifier Fairness Audit",
            option_widget=FairnessOptionsWidget(
                metric_names,
                sg.available_cohort_groups,
                fairness_ratio=0.25,
                model_options_widget=model_options_widget,
                default_metrics=["Accuracy", "Sensitivity", "Specificity", "PPV"],
            ),
            plot_function=binary_metrics_fairness_table,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot"""
        args = (
            self.metric_generator,
            list(self.option_widget.metrics),
            self.option_widget.cohorts,
            self.option_widget.fairness_ratio,
            self.option_widget.model_options.target,
            self.option_widget.model_options.score,
            self.option_widget.model_options.thresholds["Score Threshold"],
        )
        kwargs = {"per_context": self.option_widget.model_options.group_scores}
        return args, kwargs


# endregion
