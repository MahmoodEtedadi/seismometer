import logging
from typing import Optional

import traitlets
from IPython.display import HTML
from ipywidgets import Box, Dropdown, Layout, ValueWidget, VBox
from pandas import isna

from seismometer.controls.decorators import disk_cached_html_segment
from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import DisjointSelectionListsWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title
from seismometer.data import metric_apis
from seismometer.html import template
from seismometer.plot.mpl._ux import MAX_CATEGORY_SIZE
from seismometer.plot.mpl.likert import likert_plot
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


class OrdinalCategoricalSinglePlot:
    def __init__(
        self,
        metric_col: str,
        plot_type: str = "Likert Plot",
        cohort_dict: dict[str, tuple] = None,
        title: str = None,
    ):
        """
        Initializes the OrdinalCategoricalSinglePlot class.

        Parameters
        ----------
        metric_col : str
            The metric column to be plotted.
        plot_type : str, optional
            Type of plot to generate, by default "Likert Plot".
        cohort_dict : dict[str, tuple]
            Dictionary defining the cohort filter, by default None.
        title : Optional[str], optional
            Title of the plot, by default None.
        """
        self.metric_col = metric_col
        self.plot_type = plot_type
        self.title = title

        sg = Seismogram()
        self.cohort_col = next(iter(cohort_dict))
        self.cohort_values = list(cohort_dict[self.cohort_col])
        # Select columns, avoiding duplicates for Polars compatibility
        cols = [self.cohort_col, self.metric_col] if self.cohort_col != self.metric_col else [self.cohort_col]
        self.dataframe = sg.dataframe[cols]
        self.censor_threshold = sg.censor_threshold

        self.plot_functions = self.initialize_plot_functions()

        self.values = self._extract_metric_values()

        self.recorder = metric_apis.OpenTelemetryRecorder(
            metric_names=[sg.metrics[metric_col].display_name], name=self.plot_type
        )

    def _extract_metric_values(self):
        """
        Extracts the unique metric values for the provided metric column.
        """
        sg = Seismogram()
        if self.metric_col in sg.metrics:
            values = sg.metrics[self.metric_col].metric_details.values
        if values:
            return values
        values = sorted(self.dataframe[self.metric_col].unique())
        values = [value for value in values if not isna(value)]
        logger.warning(
            f"Metric values for metric {self.metric_col} are not provided. "
            + f"Using values from the corresponding dataframe column: {values}."
        )
        return values

    @classmethod
    def initialize_plot_functions(cls):
        """
        Initializes the plot functions.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping plot types to their corresponding functions.
        """
        return {
            "Likert Plot": cls.plot_likert,
        }

    def plot_likert(self):
        """
        Generates a Likert plot to show the distribution of values across the provided cohort groups
        for the metric column.

        Returns
        -------
        Optional[SVG]
            The SVS object corresponding to the generated Likert plot.
        """
        df = self._count_cohort_group_values()
        return likert_plot(df=df) if not df.empty else None

    def _count_cohort_group_values(self):
        """
        Counts occurrences of each unique value for each cohort group in the metric column.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the counts of each unique value in the metric column grouped by cohort.
        """
        import polars as pl

        # Group by and count
        df = self.dataframe.group_by([self.cohort_col, self.metric_col]).len(name="count")

        # Pivot to wide format
        df = df.pivot(on=self.metric_col, index=self.cohort_col, values="count").fill_null(0)

        # Filter to selected cohort values and maintain order
        df = df.filter(pl.col(self.cohort_col).is_in(self.cohort_values))

        # Sort by cohort values in the order they appear in self.cohort_values
        # Create a mapping for ordering
        order_map = {v: i for i, v in enumerate(self.cohort_values)}
        df = (
            df.with_columns(
                pl.col(self.cohort_col)
                .map_elements(lambda x: order_map.get(x, 999), return_dtype=pl.Int64)
                .alias("_order")
            )
            .sort("_order")
            .drop("_order")
        )

        # Add missing columns with zero values
        missing = [v for v in self.values if v not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(0).alias(col) for col in missing])
            logger.warning(f"The following metric values are missing: {missing}")

        # Select available values and cast to int
        available_values = [v for v in self.values if v in df.columns]
        df = df.select([self.cohort_col] + available_values)
        for col in available_values:
            df = df.with_columns(pl.col(col).cast(pl.Int64))

        # Filter by censor threshold (sum across row)
        row_sums = pl.sum_horizontal([pl.col(c) for c in available_values])
        df = df.filter(row_sums >= self.censor_threshold)

        # Reverse order
        df = df.reverse()

        # Instrument name (in other words, type of metric we are logging)
        instr_name = Seismogram().metrics[self.metric_col].display_name
        for row in df.iter_rows(named=True):
            cohort_value = row[self.cohort_col]
            metrics_dict = {k: v for k, v in row.items() if k != self.cohort_col}
            self.recorder.populate_metrics(
                attributes={self.cohort_col: cohort_value}, metrics={instr_name: metrics_dict}
            )

        # Convert to pandas with proper index structure for likert_plot
        df_pandas = df.to_pandas().set_index(self.cohort_col)
        df_pandas.columns.name = self.metric_col
        return df_pandas

    def generate_plot(self):
        """
        Generates the plot based on the specified plot type.

        Returns
        -------
        HTML
            The HTML object representing the generated plot figure.

        Raises
        ------
        ValueError
            If the specified plot type is unknown.
        """
        if self.plot_type not in self.plot_functions:
            raise ValueError(f"Unknown plot type: {self.plot_type}")
        if len(self.dataframe) < self.censor_threshold:
            return template.render_censored_plot_message(self.censor_threshold)
        svg = self.plot_functions[self.plot_type](self)
        return (
            template.render_title_with_image(self.title, svg)
            if svg is not None
            else template.render_censored_plot_message(self.censor_threshold)
        )


# region Plots Wrapper


@disk_cached_html_segment
def ordinal_categorical_single_col_plot(
    metric_col: str,
    cohort_dict: dict[str, tuple],
    *,
    title: Optional[str] = None,
) -> HTML:
    """
    Generates a Likert plot to show distribution of values across selected cohort groups for
    the provided metric column.

    Parameters
    ----------
    metric_col : str
        The metric column to be plotted.
    cohort_dict : dict[str, tuple]
        Dictionary defining the cohort groups to consider. Note that cohort_dict has only one key (e.g., Age),
        the cohort attribute to be studied further.
    title : Optional[str], optional
        Title of the plot, by default None.

    Returns
    -------
    HTML
        HTML object corresponding to the figure generated by the plot.
    """
    plot = OrdinalCategoricalSinglePlot(
        metric_col,
        plot_type="Likert Plot",
        cohort_dict=cohort_dict,
        title=title,
    )
    return plot.generate_plot()


# endregion
# region Plot Controls


class ExploreSingleCategoricalPlots(ExplorationWidget):
    def __init__(self, group_key: Optional[str] = None, title: Optional[str] = None):
        """
        Initializes the ExploreSingleCategoricalPlots class.

        Parameters
        ----------
        title : Optional[str], optional
            Title of the plot, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.title = title

        metrics = (
            [
                metric
                for metric in sg.metric_groups[group_key]
                if metric in sg.get_ordinal_categorical_metrics(MAX_CATEGORY_SIZE)
            ]
            if group_key
            else sg.get_ordinal_categorical_metrics(MAX_CATEGORY_SIZE)
        )

        super().__init__(
            title="Plot Cohort Distribution",
            option_widget=CategoricalFeedbackSingleColumnOptionsWidget(
                metrics,
                cohort_groups=sg.available_cohort_groups,
                title=title,
            ),
            plot_function=ordinal_categorical_single_col_plot,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the ordinal categorical single column plot."""
        cohort_dict = {self.option_widget.cohort_list[0]: tuple(self.option_widget.cohort_list[1])}
        args = (
            self.option_widget.metric_col,
            cohort_dict,
        )
        kwargs = {"title": self.option_widget.dynamic_title}
        return args, kwargs


class CategoricalFeedbackSingleColumnOptionsWidget(Box, ValueWidget, traitlets.HasTraits):
    value = traitlets.Dict(help="The selected values for the ordinal categorical options.")

    def __init__(
        self,
        metrics: list[str],
        cohort_groups: dict[str, list[str]],
        *,
        model_options_widget=None,
        title: str = None,
    ):
        """
        Initializes the CategoricalFeedbackSingleColumnOptionsWidget class.

        Parameters
        ----------
        metrics : list[str]
            List of available metric columns.
        cohort_groups : dict[str, list[str]]
            Dictionary defining the cohort attribute and the corresponding cohort groups to consider.
        model_options_widget : Optional[widget], optional
            Additional widget options if needed, by default None.
        title : str, optional
            Title of the plot, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.model_options_widget = model_options_widget
        self.title = title
        self.dynamic_title = title

        metric_options = metrics or sg.get_ordinal_categorical_metrics(MAX_CATEGORY_SIZE)
        self.metric_display_name_to_source = {sg.metrics[metric].display_name: metric for metric in metric_options}
        metric_options = list(self.metric_display_name_to_source.keys())
        self._metric_col = Dropdown(
            options=metric_options,
            value=metric_options[0],
            description="Metric",
            style=WIDE_LABEL_STYLE,
        )

        self._cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter")

        self._metric_col.observe(self._on_value_changed, names="value")
        self._cohort_list.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Plot Options"),
            self._metric_col,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        vbox_layout = Layout(align_items="flex-end", flex="0 0 auto")

        vbox = VBox(children=v_children, layout=vbox_layout)

        super().__init__(
            children=[vbox, self._cohort_list],
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
        self._metric_col.disabled = value
        self._cohort_list.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        title = (self.title or "").strip()
        self.dynamic_title = f"{self.title}: {self._metric_col.value}" if title else self._metric_col.value
        new_value = {
            "metric_col": self._metric_col.value,
            "cohort_list": self._cohort_list.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def metric_col(self):
        return self.metric_display_name_to_source[self._metric_col.value]

    @property
    def cohort_list(self):
        return self._cohort_list.value


# endregion
