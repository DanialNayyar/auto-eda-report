import pandas as pd
import pytest
 
from custom_eda.correlations import get_strong_feature_correlations, summary_of_correlations


def test_get_strong_feature_correlations_detects_positive_and_negative_correlations():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [2, 4, 6, 8, 10],
        "feature_c": [5, 4, 3, 2, 1]
    })

    result = get_strong_feature_correlations(
        data_frame=df,
        correlation_threshold=0.9
    )

    assert result["feature_a"]["feature_b"] == pytest.approx(1.0)
    assert result["feature_a"]["feature_c"] == pytest.approx(-1.0)

  
    assert result["feature_b"]["feature_a"] == pytest.approx(1.0)
    assert result["feature_c"]["feature_a"] == pytest.approx(-1.0)



def test_get_strong_feature_correlations_excludes_weak_correlations():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5, 6],
        "feature_b": [2, 4, 6, 8, 10, 12],
        "feature_c": [1, 4, 2, 6, 3, 5]
    })

    result = get_strong_feature_correlations(
        data_frame=df,
        correlation_threshold=0.9
    )

    assert "feature_b" in result["feature_a"]
    assert "feature_c" not in result["feature_a"]

    # feature_c has no correlation meeting the threshold,
    # so it should not receive its own result entry.
    assert "feature_c" not in result


def test_get_strong_feature_correlations_excludes_self_correlations():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4],
        "feature_b": [2, 4, 6, 8]
    })

    result = get_strong_feature_correlations(
        data_frame=df,
        correlation_threshold=0.9
    )

    assert "feature_a" not in result["feature_a"]
    assert "feature_b" not in result["feature_b"]


def test_get_strong_feature_correlations_supports_spearman():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [1, 4, 9, 16, 25]
    })

    result = get_strong_feature_correlations(
        data_frame=df,
        correlation_threshold=0.99,
        method="spearman"
    )

    assert result["feature_a"]["feature_b"] == pytest.approx(1.0)
    assert result["feature_b"]["feature_a"] == pytest.approx(1.0)


def test_get_strong_feature_correlations_supports_kendall():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [10, 8, 6, 4, 2]
    })

    result = get_strong_feature_correlations(
        data_frame=df,
        correlation_threshold=0.9,
        method="kendall"
    )

    assert result["feature_a"]["feature_b"] == pytest.approx(-1.0)
    assert result["feature_b"]["feature_a"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "invalid_threshold",
    [0, -0.1, 1.1]
)
def test_get_strong_feature_correlations_rejects_invalid_thresholds(
    invalid_threshold
):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [2, 4, 6]
    })

    with pytest.raises(ValueError):
        get_strong_feature_correlations(
            data_frame=df,
            correlation_threshold=invalid_threshold
        )


def test_get_strong_feature_correlations_rejects_invalid_method():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [2, 4, 6]
    })

    with pytest.raises(ValueError):
        get_strong_feature_correlations(
            data_frame=df,
            correlation_threshold=0.8,
            method="invalid_method"
        )


def test_get_strong_feature_correlations_rejects_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        get_strong_feature_correlations(
            data_frame=df,
            correlation_threshold=0.8
        )


def test_get_strong_feature_correlations_rejects_dataframe_without_usable_columns():
    df = pd.DataFrame({
        "city": ["London", "Leeds", "Liverpool"],
        "category": ["A", "B", "C"]
    })

    with pytest.raises(ValueError):
        get_strong_feature_correlations(
            data_frame=df,
            correlation_threshold=0.8
        )


def test_get_strong_feature_correlations_returns_empty_dict_for_one_numeric_column():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["London", "Leeds", "Liverpool"]
    })

    result = get_strong_feature_correlations(
        data_frame=df,
        correlation_threshold=0.8
    )

    assert result == {}


def test_get_strong_feature_correlations_includes_boolean_columns():
    df = pd.DataFrame({
        "numeric_feature": [0, 1, 0, 1],
        "boolean_feature": [False, True, False, True],
        "category": ["A", "B", "C", "D"]
    })

    result = get_strong_feature_correlations(
        data_frame=df,
        correlation_threshold=0.9
    )

    assert result["numeric_feature"]["boolean_feature"] == pytest.approx(1.0)
    assert result["boolean_feature"]["numeric_feature"] == pytest.approx(1.0)




def test_summary_of_correlations_returns_expected_summary():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [2, 4, 6, 8, 10],
        "category": ["A", "B", "C", "D", "E"]
    })

    result = summary_of_correlations(
        df=df,
        threshold=0.9
    )

    assert result["correlation_method"] == "pearson"
    assert result["correlation_threshold"] == 0.9

    assert result["columns_analysed"] == [
        "feature_a",
        "feature_b"
    ]
    assert result["number_of_columns_analysed"] == 2

    assert "category" not in result["columns_analysed"]

    assert result["correlations"]["feature_a"]["feature_a"] == pytest.approx(1.0)
    assert result["correlations"]["feature_a"]["feature_b"] == pytest.approx(1.0)
    assert result["correlations"]["feature_b"]["feature_a"] == pytest.approx(1.0)

    assert result["strongest_correlations"]["feature_a"]["feature_b"] == pytest.approx(1.0)
    assert result["strongest_correlations"]["feature_b"]["feature_a"] == pytest.approx(1.0)


def test_summary_of_correlations_can_exclude_strongest_correlations():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4],
        "feature_b": [2, 4, 6, 8]
    })

    result = summary_of_correlations(
        df=df,
        threshold=0.9,
        return_strongest=False
    )

    assert result["strongest_correlations"] is None

    # The complete correlation matrix should still be returned.
    assert result["correlations"]["feature_a"]["feature_b"] == pytest.approx(1.0)


def test_summary_of_correlations_normalises_method_name():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [1, 4, 9, 16, 25]
    })

    result = summary_of_correlations(
        df=df,
        threshold=0.9,
        method="  SPEARMAN  "
    )

    assert result["correlation_method"] == "spearman"
    assert result["correlations"]["feature_a"]["feature_b"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "method",
    ["pearson", "spearman", "kendall"]
)
def test_summary_of_correlations_supports_allowed_methods(method):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [2, 4, 6, 8, 10]
    })

    result = summary_of_correlations(
        df=df,
        threshold=0.9,
        method=method
    )

    assert result["correlation_method"] == method
    assert result["correlations"]["feature_a"]["feature_b"] == pytest.approx(1.0)


def test_summary_of_correlations_rejects_invalid_method():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [2, 4, 6]
    })

    with pytest.raises(
        ValueError,
        match="Method must be pearson, spearman, or kendall"
    ):
        summary_of_correlations(
            df=df,
            threshold=0.8,
            method="invalid_method"
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [0, -0.1, 1.1]
)
def test_summary_of_correlations_rejects_invalid_threshold(
    invalid_threshold
):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [2, 4, 6]
    })

    with pytest.raises(
        ValueError,
        match="Threshold must be greater than 0"
    ):
        summary_of_correlations(
            df=df,
            threshold=invalid_threshold
        )


def test_summary_of_correlations_rejects_dataframe_without_usable_columns():
    df = pd.DataFrame({
        "city": ["London", "Leeds", "Liverpool"],
        "category": ["A", "B", "C"]
    })

    with pytest.raises(
        ValueError,
        match="No numerical or boolean columns"
    ):
        summary_of_correlations(
            df=df,
            threshold=0.8
        )


def test_summary_of_correlations_returns_empty_dict_for_one_usable_column():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["London", "Leeds", "Liverpool"]
    })

    result = summary_of_correlations(
        df=df,
        threshold=0.8
    )

    assert result == {}


def test_summary_of_correlations_includes_boolean_columns():
    df = pd.DataFrame({
        "numeric_feature": [0, 1, 0, 1],
        "boolean_feature": [False, True, False, True],
        "category": ["A", "B", "C", "D"]
    })

    result = summary_of_correlations(
        df=df,
        threshold=0.9
    )

    assert result["columns_analysed"] == [
        "numeric_feature",
        "boolean_feature"
    ]
    assert result["number_of_columns_analysed"] == 2
    assert "category" not in result["correlations"]

    assert result["correlations"]["numeric_feature"]["boolean_feature"] == pytest.approx(1.0)