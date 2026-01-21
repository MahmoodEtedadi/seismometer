from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from ipywidgets import HTML

import seismometer.table.fairness as undertest
from seismometer.data.performance import BinaryClassifierMetricGenerator, MetricGenerator

# ---- Fixtures ----


def sample_data():
    # Create DataFrame with string type for Count column to allow mixed values
    df = pd.DataFrame(
        {
            "Cohort": ["Last", "First", "Middle", "Last", "First", "Middle", "Last", "First", "Middle"],
            "Class": ["L1", "Fn", "M3", "L4", "F5", "M?", "L7", "F8", "M9"],
            "Count": [1, np.nan, 3, 4, 5, undertest.FairnessIcons.UNKNOWN.value, 7, 8, 9],
        }
    )
    return df


def large_dataset_data():
    return pd.DataFrame(
        {
            "Cohort": ["Last", "First", "Middle"] * 10 + ["First", "Middle", "Middle"] * 10 + ["Middle"] * 30,
            "Category": ["L1", "F3", "M3", "L4", "F5", "M8", "L7", "F8", "M9"] * 10,
            "Number": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 10,
        }
    )


# ---- Test Classes ----


class TestSortKeys:
    def test_sort_keys(self):
        # Create Polars DataFrame directly with mixed Count column (use Object type)
        data = pl.DataFrame(
            {
                "Cohort": ["Last", "First", "Middle", "Last", "First", "Middle", "Last", "First", "Middle"],
                "Class": ["L1", "Fn", "M3", "L4", "F5", "M?", "L7", "F8", "M9"],
                "Count": pl.Series(
                    [1, None, 3, 4, 5, undertest.FairnessIcons.UNKNOWN.value, 7, 8, 9], dtype=pl.Object
                ),
            }
        )
        data = undertest.sort_fairness_table(data, ["First", "Middle", "Last"])
        result_pd = data.to_pandas()

        # Verify cohort grouping
        cohort_list = result_pd["Cohort"].tolist()
        assert cohort_list == [
            "First",
            "First",
            "First",
            "Middle",
            "Middle",
            "Middle",
            "Last",
            "Last",
            "Last",
        ]

        # Verify class ordering (sorted by cohort, then by count descending, with None/emoji at end of each group)
        # First group: None sorts first (as NaN), then 8, then 5
        # Middle group: 9, 3, then emoji
        # Last group: 7, 4, 1
        assert result_pd["Class"].tolist() == ["Fn", "F8", "F5", "M9", "M3", "M?", "L7", "L4", "L1"]

        # Check individual count values
        count_list = result_pd["Count"].tolist()
        assert pd.isna(count_list[0]) or count_list[0] is None  # None from "Fn"
        assert count_list[1] == 8  # "F8"
        assert count_list[2] == 5  # "F5"
        assert count_list[3] == 9  # "M9"
        assert count_list[4] == 3  # "M3"
        assert count_list[5] == "❔"  # "M?"
        assert count_list[6] == 7  # "L7"
        assert count_list[7] == 4  # "L4"
        assert count_list[8] == 1  # "L1"


class TestFairnessTable:
    def test_fairness_table_filters_values_small(self):
        # Create a simple dataframe without mixed types
        data = pl.DataFrame(
            {
                "Cohort": ["Last", "First", "Middle", "Last", "First", "Middle", "Last", "First", "Middle"],
                "Class": ["L1", "Fn", "M3", "L4", "F5", "M?", "L7", "F8", "M9"],
            }
        )
        fake_metrics = MetricGenerator(["M1", "M2", "M3"], lambda x, names: {"M1": 1, "M2": 2, "M3": 3})
        table = undertest.fairness_table(data, fake_metrics, ["M1", "M2"], 0.1, {"Cohort": ["First", "Middle"]})
        assert "Last" not in table.value
        assert "M3" not in table.value

    def test_fairness_table_filters_values_large(self):
        data = pl.from_pandas(large_dataset_data())

        # Adapt the metric function to work with Polars
        def polars_metric_fn(x, names):
            x_pd = x.to_pandas() if hasattr(x, "to_pandas") else x
            return {"M1": x_pd.Number.mean(), "M2": x_pd.Category.str.count("F8").sum(), "M3": 3}

        fake_metrics = MetricGenerator(["M1", "M2", "M3"], polars_metric_fn)
        table = undertest.fairness_table(data, fake_metrics, ["M1", "M2"], 0.1, {"Cohort": ["First", "Middle"]})
        assert "Last" not in table.value
        assert "M3" not in table.value
        assert "60" in table.value
        assert "🔹  5.43" in table.value
        assert "🔻  3.00" in table.value
        assert "🔻  4.35" in table.value
        assert "🔹  7.00" in table.value


class TestFairnessIcons:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.5, undertest.FairnessIcons.CRITICAL_LOW),
            (0.75, undertest.FairnessIcons.WARNING_LOW),
            (0.9, undertest.FairnessIcons.GOOD),
            (1.0, undertest.FairnessIcons.DEFAULT),
            (1.1, undertest.FairnessIcons.GOOD),
            (1.26, undertest.FairnessIcons.WARNING_HIGH),
            (1.71, undertest.FairnessIcons.CRITICAL_HIGH),
            (None, undertest.FairnessIcons.UNKNOWN),
        ],
    )
    def test_values(self, value, expected):
        assert undertest.FairnessIcons.get_fairness_icon(value) == expected


class TestFairnessLegend:
    def test_get_fairness_legend_contains_expected_text(self):
        legend = undertest.FairnessIcons.get_fairness_legend(limit=0.2, open=False, censor_threshold=5)
        legend_str = str(legend)
        assert "Within 20.00%" in legend_str
        assert "fewer than 5 observations" in legend_str


class TestFairnessTableValidation:
    def test_fairness_table_raises_value_error_on_bad_inputs(self):
        df = pd.DataFrame({"group": ["A", "B"], "value": [1, 2]})
        dummy_metric = MetricGenerator(["M1"], lambda x, names: {"M1": 1})

        with pytest.raises(ValueError, match="Fairness ratio must be greater than 0"):
            undertest.fairness_table(df, dummy_metric, ["M1"], 0.0, {"group": ("A", "B")})

        with pytest.raises(ValueError, match="No cohorts provided"):
            undertest.fairness_table(df, dummy_metric, ["M1"], 0.25, None)

    def test_fairness_table_censors_small_groups(self):
        df = pl.DataFrame(
            {
                "group": ["A", "B", "A", "B"],
                "val": [1, 2, 1, 2],
            }
        )
        metric_fn = MetricGenerator(["M1"], lambda x, names: {"M1": 1.0})
        cohort_dict = {"group": ("A", "B")}

        result: HTML = undertest.fairness_table(df, metric_fn, ["M1"], 0.25, cohort_dict, censor_threshold=10)
        assert "❔" in result.value or "--" in result.value


class TestFairnessWrappers:
    def test_binary_metrics_fairness_table_runs(self, monkeypatch):
        sg_mock = Mock()
        sg_mock.dataframe = pl.DataFrame({"group": ["A", "B"], "val": [1, 2]})
        sg_mock.entity_keys = []
        sg_mock.predict_time = None
        sg_mock.censor_threshold = 10
        sg_mock.event_aggregation_method.return_value = "mean"
        monkeypatch.setattr("seismometer.seismogram.Seismogram", lambda: sg_mock)

        gen = BinaryClassifierMetricGenerator()
        with patch.object(BinaryClassifierMetricGenerator, "metric_names", new_callable=PropertyMock) as mock_metrics:
            mock_metrics.return_value = ["M1"]
            html_result = undertest.binary_metrics_fairness_table(
                gen, ["M1"], {"group": ("A", "B")}, 0.25, "target", "score", 0.5
            )
        assert isinstance(html_result, HTML)

    def test_custom_metrics_fairness_table_runs(self, monkeypatch):
        sg_mock = Mock()
        sg_mock.dataframe = pl.DataFrame({"group": ["A", "B"], "val": [1, 2]})
        sg_mock.available_cohort_groups = {"group": ("A", "B")}
        sg_mock.censor_threshold = 10
        monkeypatch.setattr("seismometer.seismogram.Seismogram", lambda: sg_mock)

        gen = MetricGenerator(["M1"], lambda x, names: {"M1": 1})
        html_result = undertest.custom_metrics_fairness_table(gen, ["M1"], None, 0.25)
        assert isinstance(html_result, HTML)


class TestFairnessOptionsWidget:
    @patch("seismometer.seismogram.Seismogram")
    def test_fairness_options_widget_value_behavior(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        metric_names = ("M1", "M2")
        cohort_dict = {"group": ("A", "B")}
        widget = undertest.FairnessOptionsWidget(metric_names, cohort_dict, fairness_ratio=0.3)

        # Initial state: cohort_list is empty
        val = widget.value
        assert list(val["metric_list"]) == list(metric_names)
        assert val["cohort_list"] == {}  # no selection yet
        assert widget.cohorts == cohort_dict  # fallback works

        # Simulate user selecting both values
        widget.cohort_list.value = {"group": ("A", "B")}
        widget._on_value_changed()

        updated_val = widget.value
        assert updated_val["cohort_list"] == {"group": ("A", "B")}  # now it reflects user input

        # Test enabling/disabling
        widget.disabled = True
        assert widget.metric_list.disabled
        assert widget.cohort_list.disabled
        assert widget.fairness_slider.disabled

        widget.disabled = False
        assert not widget.metric_list.disabled


class TestExplorationFairnessWidget:
    @patch("seismometer.seismogram.Seismogram")
    def test_initialization_sets_up_components(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        fake_seismo.available_cohort_groups = {"gender": ("M", "F"), "age": ("young", "old")}

        metrics = MetricGenerator(["Accuracy", "Sensitivity"], lambda df, names: {k: 1 for k in names})

        widget = undertest.ExplorationFairnessWidget(metrics)

        assert widget.metrics_generator == metrics
        assert isinstance(widget.option_widget, undertest.FairnessOptionsWidget)
        assert widget.option_widget.fairness_ratio == 0.25
        assert widget.plot_function == undertest.custom_metrics_fairness_table

    @patch("seismometer.seismogram.Seismogram")
    def test_generate_plot_args_returns_expected_values(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        fake_seismo.available_cohort_groups = {"gender": ("M", "F")}

        metrics = MetricGenerator(["AUC", "Accuracy"], lambda df, names: {k: 1 for k in names})
        widget = undertest.ExplorationFairnessWidget(metrics)

        # Simulate user selections
        widget.option_widget.metric_list.value = ["AUC"]
        widget.option_widget.cohort_list.value = {"gender": ("M",)}
        widget.option_widget.fairness_slider.value = 0.4

        args, kwargs = widget.generate_plot_args()

        # Validate returned args
        assert args[0] == metrics
        assert args[1] == ["AUC"]
        assert args[2] == {"gender": ("M",)}
        assert args[3] == 0.4
        assert kwargs == {}
