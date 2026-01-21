from typing import Any, Optional

from IPython.display import HTML, display

from seismometer.controls.decorators import disk_cached_html_segment
from seismometer.controls.explore import ExplorationWidget  # noqa:
from seismometer.controls.explore import (
    ExplorationCohortOutcomeInterventionEvaluationWidget,
    ExplorationCohortSubclassEvaluationWidget,
    ExplorationMetricWidget,
    ExplorationModelSubgroupEvaluationWidget,
    ExplorationScoreComparisonByCohortWidget,
    ExplorationSubpopulationWidget,
    ExplorationTargetComparisonByCohortWidget,
)
from seismometer.core.decorators import export
from seismometer.data import pandas_helpers as pdh
from seismometer.data.performance import BinaryClassifierMetricGenerator
from seismometer.html import template
from seismometer.seismogram import Seismogram

from .plots import (
    plot_binary_classifier_metrics,
    plot_cohort_evaluation,
    plot_cohort_group_histograms,
    plot_cohort_lead_time,
    plot_intervention_outcome_timeseries,
    plot_model_evaluation,
    plot_model_score_comparison,
    plot_model_target_comparison,
)


# region Exploration Widgets
@export
class ExploreSubgroups(ExplorationSubpopulationWidget):
    """
    Explore the models base statistics based on a selected subpopulation.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Subpopulation Statistics", cohort_list_details)


@export
class ExploreModelEvaluation(ExplorationModelSubgroupEvaluationWidget):
    """
    Exploration widget for model evaluation, showing model performance for a specific subpopulation.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/ppv, and a histogram.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Model Performance", plot_model_evaluation)


@export
class ExploreModelScoreComparison(ExplorationScoreComparisonByCohortWidget):
    """
    Exploration widget for model evaluation, showing model performance for a specific subpopulation.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/ppv, and a histogram.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Model Score Comparison", plot_model_score_comparison)


@export
class ExploreModelTargetComparison(ExplorationTargetComparisonByCohortWidget):
    """
    Exploration widget for model target evaluation, showing model performance for a specific subpopulation.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/PPV, and a histogram.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Model Target Comparison", plot_model_target_comparison)


@export
class ExploreCohortEvaluation(ExplorationCohortSubclassEvaluationWidget):
    """
    Exploration widget for cohort evaluation, showing model performance across thresholds and cohort subgroups.

    Creates a 2x3 grid of individual performance metrics across cohorts.

    Plots include Sensitivity, Flag Rate, PPV, Specificity, NPV vs Thresholds.
    Includes a legend with cohort size.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Cohort Group Performance", plot_cohort_evaluation)


@export
class ExploreCohortHistograms(ExplorationCohortSubclassEvaluationWidget):
    """
    Exploration widget to show the true positives and negative by model score.

    Shows a distribution of scores for each category in a cohort group.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__(
            "Cohort Group Score Histograms",
            plot_cohort_group_histograms,
            threshold_handling=None,
            ignore_grouping=True,
        )


@export
class ExploreCohortLeadTime(ExplorationCohortSubclassEvaluationWidget):
    """
    Exploration widget for the lead time between a model prediction and an event of interest.

    Shows the amount of lead time for each category in the cohort group.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__(
            "Leadtime Analysis",
            plot_cohort_lead_time,
            threshold_handling="min",
            ignore_grouping=True,
        )


@export
class ExploreBinaryModelMetrics(ExplorationMetricWidget):
    """
    Explore the models performance metrics based on a selected metric.
    """

    def __init__(self, rho: Optional[float] = None):
        """
        Passes the plot function to the superclass.

        Parameters
        ----------
        rho: float between 0 and 1
           Probability of a treatment being effective
        """
        metric_generator = BinaryClassifierMetricGenerator(rho=rho)
        super().__init__("Model Metric Evaluation", metric_generator, plot_binary_classifier_metrics)


@export
class ExploreCohortOutcomeInterventionTimes(ExplorationCohortOutcomeInterventionEvaluationWidget):
    """
    Exploration widget for viewing rates of interventions and outcomes across categories in a cohort group.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Outcome / Intervention Analysis", plot_intervention_outcome_timeseries)


# endregion


@export
def cohort_list():
    """
    Displays an exhaustive list of available cohorts for analysis.
    """
    sg = Seismogram()
    from ipywidgets import Output, VBox

    from seismometer.controls.selection import MultiSelectionListWidget
    from seismometer.controls.styles import BOX_GRID_LAYOUT

    options = sg.available_cohort_groups

    comparison_selections = MultiSelectionListWidget(
        options,
        title="Cohort",
        show_all=True,
        hierarchies=sg.cohort_hierarchies,
        hierarchy_combinations=sg.cohort_hierarchy_combinations,
    )
    output = Output()

    def on_widget_value_changed(*args):
        with output:
            display("Recalculating...", clear=True)
            html = cohort_list_details(comparison_selections.value)
            display(html, clear=True)

    comparison_selections.observe(on_widget_value_changed, "value")

    # get initial value
    on_widget_value_changed()

    return VBox(children=[comparison_selections, output], layout=BOX_GRID_LAYOUT)


@disk_cached_html_segment
@export
def cohort_list_details(cohort_dict: dict[str, tuple[Any]]) -> HTML:
    """
    Generates an HTML table of cohort details.

    Parameters
    ----------
    cohort_dict : dict[str, tuple[Any]]
        dictionary of cohort columns and values used to subselect a population for evaluation

    Returns
    -------
    HTML
        able indexed by targets, with counts of unique entities, and mean values of the output columns.
    """
    from seismometer.data.filter import filter_rule_from_cohort_dictionary

    sg = Seismogram()
    cfg = sg.config
    target_cols = [pdh.event_value(x) for x in cfg.targets]
    intervention_cols = [pdh.event_value(x) for x in cfg.interventions]
    outcome_cols = [pdh.event_value(x) for x in cfg.outcomes]

    rule = filter_rule_from_cohort_dictionary(cohort_dict)
    import polars as pl

    data = rule.filter(sg.dataframe).select(
        cfg.entity_keys + cfg.output_list + intervention_cols + outcome_cols + target_cols
    )
    cohort_count = data[sg.entity_keys[0]].n_unique()
    if cohort_count < sg.censor_threshold:
        return template.render_censored_plot_message(sg.censor_threshold)

    # Identify float columns
    float_cols = [col for col in intervention_cols + outcome_cols if data.schema[col] in [pl.Float32, pl.Float64]]

    # Build aggregation expressions
    agg_exprs = []
    agg_col_names = []

    # Mean for float columns
    for col in float_cols:
        agg_exprs.append(pl.col(col).mean().alias(pdh.event_name(col)))
        agg_col_names.append(pdh.event_name(col))

    # Entity stats
    agg_exprs.append(pl.col(cfg.entity_id).n_unique().alias(f"Unique {cfg.entity_id}"))
    agg_col_names.append(f"Unique {cfg.entity_id}")
    agg_exprs.append(pl.col(cfg.entity_id).len().alias(f"{cfg.entity_id} Count"))
    agg_col_names.append(f"{cfg.entity_id} Count")

    # Context stats if available
    if cfg.context_id is not None:
        agg_exprs.append(pl.col(cfg.context_id).n_unique().alias(f"Unique {cfg.context_id}"))
        agg_col_names.append(f"Unique {cfg.context_id}")

    groupstats = data.group_by(target_cols, maintain_order=True).agg(agg_exprs)

    # Build HTML table directly from Polars DataFrame
    html_parts = ['<table border="1" class="dataframe">']

    # Header row
    html_parts.append('<thead><tr style="text-align: right;">')
    for col in target_cols:
        html_parts.append(f"<th>{col}</th>")
    for col in agg_col_names:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead>")

    # Data rows
    html_parts.append("<tbody>")
    for row in groupstats.iter_rows(named=True):
        html_parts.append("<tr>")
        for col in target_cols:
            html_parts.append(f"<td>{row[col]}</td>")
        for col in agg_col_names:
            val = row[col]
            if isinstance(val, float):
                html_parts.append(f"<td>{val:.4f}</td>")
            else:
                html_parts.append(f"<td>{val}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    html_table = "".join(html_parts)
    title = "Summary"
    return template.render_title_message(title, html_table)


# endregion
