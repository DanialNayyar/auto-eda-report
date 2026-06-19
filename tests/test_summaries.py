import pandas as pd
import pytest

from custom_eda.summaries import dataset_overview, summary_of_data_types, summary_of_missing_values, summary_of_duplicates, summary_of_numerical_data, summary_of_categorical_data, summaries_of_targets

def test_dataset_overview_returns_correct_strucutre():
    df = pd.DataFrame({
        "age": [20, 50, 30],
        "city": ["London", "New York", "Liverpool"] 
    })

    result = dataset_overview(df=df)


    assert result["shape"] == (3,2)
    assert result["column_names"] == ["age", "city"]
    assert result["number_of_rows"] == 3
    assert result["number_of_columns"] == 2
    assert result["index_type"] == "RangeIndex"
    assert result["memory_usage_mb"] > 0



def test_summary_of_data_types():
    df = pd.DataFrame({
        "age": [20, 50],
        "city": ["London", "New York"],
        "height": [1.7, 1.8],
        "passed": [True, False],
        "date":pd.to_datetime(["2026-01-01", "2026-01-02"])
    
    })

    result = summary_of_data_types(df=df)

    assert result["numeric_cols"] == ["age", "height"]
    assert result["categorical_object"] == ["city"]
    assert result["boolean_cols"] == ["passed"]
    assert result["other"] == []
    assert result["datetime_col"] == ["date"]



def test_summary_of_missing_values():
    df = pd.DataFrame({

        "age": [20, None, 40],
        "city": ["London", None, "Leeds"],
        "passed": [True, False, True]
    
    })

    result = summary_of_missing_values(df=df)

    assert result.loc["age", "missing_count"] == 1
    assert result.loc["city", "missing_count"] == 1
    assert result.loc["passed", "missing_count"] == 0        

    assert result.loc["age", "missing_percent"] == pytest.approx(33.333333)
    assert result.loc["city", "missing_percent"] == pytest.approx(33.333333)
    assert result.loc["passed", "missing_percent"] == 0





def test_summary_of_duplicates():
    df = pd.DataFrame({

        "age": [20, 40, 20],
        "city": ["London", "Leeds", "London"],
        "passed": [True, False, True]
    
    })

    result = summary_of_duplicates(df=df)

    assert result["duplicate_count"] == 1
    assert result["duplicate_percent"] == pytest.approx(33.33333)




def test_summary_of_numerical_data():
    df = pd.DataFrame({
        "age": [10, 20, None, 40],
        "score": [1.0, 2.0, 3.0, 4.0],
        "city": ["London", "Leeds", "London", "Liverpool"]
    })


    result = summary_of_numerical_data(df=df)

    assert list(result.index) == ["age", "score"]
    assert "city" not in result.index

    assert result.loc["age", "count"] == 3
    assert result.loc["age", "mean"] == pytest.approx(23.333)
    assert result.loc["age", "median"] == 20
    assert result.loc["age", "range"] == 30
    assert result.loc["age", "missing_count"] == 1
    assert result.loc["age", "missing_percentage"] == 25
    assert result.loc["age", "unique_count"] == 3
    assert result.loc["age", "unique_percentage"] == 75



    assert result.loc["score", "count"] == 4
    assert result.loc["score", "mean"] == 2.5
    assert result.loc["score", "median"] == 2.5
    assert result.loc["score", "range"] == 3
    assert result.loc["score", "missing_count"] == 0
    assert result.loc["score", "missing_percentage"] == 0


def test_summary_of_categorical_data():
    df = pd.DataFrame({
        "city": ["London", "London", "Leeds", None],
        "passed": [True, True, True, False],
        "age": [20, 30, 40, 50]
    })

    result = summary_of_categorical_data(df=df)
        
    result = result.set_index("column_name")

    assert list(result.index) == ["city", "passed"]
    assert "age" not in result.index


    assert result.loc["city", "column_type"] == "object"
    assert result.loc["city", "unique_count"] == 2
    assert result.loc["city", "unique_percentage"] == 50.0
    assert result.loc["city", "missing_count"] == 1
    assert result.loc["city", "missing_percentage"] == 25.0
    assert result.loc["city", "most_frequent"] == "London"
    assert result.loc["city", "most_frequent_count"] == 2
    assert result.loc["city", "most_frequent_percentage"] == 50.0


    assert result.loc["passed", "column_type"] == "bool"
    assert result.loc["passed", "unique_count"] == 2
    assert result.loc["passed", "missing_count"] == 0
    assert result.loc["passed", "most_frequent"] == True
    assert result.loc["passed", "most_frequent_count"] == 3
    assert result.loc["passed", "most_frequent_percentage"] == 75.0   


def test_summaries_of_targets_for_classification():
    df = pd.DataFrame({
        "diagnosis": ["benign", "malignant", "benign", None],
        "age": [40, 55, 47, 61]
    })

    result = summaries_of_targets(
        df=df,
        target_names="diagnosis",
        problem_types="classification"
    )

    diagnosis_summary = result["diagnosis"]

    assert diagnosis_summary["problem_type"] == "classification"
    assert diagnosis_summary["data_type"] == "object"

    assert diagnosis_summary["unique_count"] == 2
    assert diagnosis_summary["unique_percentage"] == 50.0

    assert diagnosis_summary["missing_count"] == 1
    assert diagnosis_summary["missing_percentage"] == 25.0

    assert diagnosis_summary["unique_values"] == ["benign", "malignant"]

    assert diagnosis_summary["class_counts"]["benign"] == 2
    assert diagnosis_summary["class_counts"]["malignant"] == 1

    assert diagnosis_summary["class_percentages"]["benign"] == pytest.approx(66.6667)
    assert diagnosis_summary["class_percentages"]["malignant"] == pytest.approx(33.333333)



def test_summaries_of_targets_for_regression():
    df = pd.DataFrame({
        "house_price": [100000, 150000, 200000, 250000],
        "city": ["London", "Leeds", "Liverpool", "Manchester"]
    })

    result = summaries_of_targets(
        df=df,
        target_names="house_price",
        problem_types="regression"
    )

    price_summary = result["house_price"]

    assert price_summary["problem_type"] == "regression"
    assert price_summary["unique_count"] == 4
    assert price_summary["unique_percentage"] == 100.0
    assert price_summary["missing_count"] == 0
    assert price_summary["missing_percentage"] == 0.0

    assert price_summary["mean"] == 175000.0
    assert price_summary["median"] == 175000.0
    assert price_summary["minimum"] == 100000.0
    assert price_summary["maximum"] == 250000.0
    assert price_summary["range"] == 150000.0

    assert price_summary["standard_deviation"] == pytest.approx(64549.7224)

    assert "class_counts" not in price_summary
    assert "class_percentages" not in price_summary


def test_summaries_of_targets_requires_target_names():
    df = pd.DataFrame({
        "diagnosis": ["benign", "malignant"]
    })

    with pytest.raises(
        ValueError,
        match="target names is empty"
    ):
        summaries_of_targets(
            df=df,
            target_names=None,
            problem_types="classification"
        )


def test_summaries_of_targets_rejects_missing_target_column():
    df = pd.DataFrame({
        "diagnosis": ["benign", "malignant"]
    })

    with pytest.raises(
        ValueError,
        match="The following are not in the dataset"
    ):
        summaries_of_targets(
            df=df,
            target_names="tumour_type",
            problem_types="classification"
        )


def test_summaries_of_targets_requires_problem_types():
    df = pd.DataFrame({
        "diagnosis": ["benign", "malignant"]
    })

    with pytest.raises(
        ValueError,
        match="problem types is empty"
    ):
        summaries_of_targets(
            df=df,
            target_names="diagnosis",
            problem_types=None
        )

def test_summaries_of_targets_rejects_mismatched_lengths():
    df = pd.DataFrame({
        "diagnosis": ["benign", "malignant"],
        "price": [100000, 200000]
    })

    with pytest.raises(
        ValueError,
        match="Problem types is not the same length as target names"
    ):
        summaries_of_targets(
            df=df,
            target_names=["diagnosis", "price"],
            problem_types=["classification"]
        )


def test_summaries_of_targets_rejects_invalid_problem_type():
    df = pd.DataFrame({
        "diagnosis": ["benign", "malignant"]
    })

    with pytest.raises(
        ValueError,
        match="The following are not in the allowed problems"
    ):
        summaries_of_targets(
            df=df,
            target_names="diagnosis",
            problem_types="clustering"
        )



def test_summaries_of_targets_handles_multiple_targets():
    df = pd.DataFrame({
        "diagnosis": ["benign", "malignant", "benign"],
        "tumour_size": [2.0, 5.0, 8.0]
    })

    result = summaries_of_targets(
        df=df,
        target_names=["diagnosis", "tumour_size"],
        problem_types=["classification", "regression"]
    )

    assert list(result.keys()) == ["diagnosis", "tumour_size"]

    assert result["diagnosis"]["problem_type"] == "classification"
    assert result["diagnosis"]["class_counts"]["benign"] == 2
    assert result["diagnosis"]["class_counts"]["malignant"] == 1

    assert result["tumour_size"]["problem_type"] == "regression"
    assert result["tumour_size"]["mean"] == 5.0
    assert result["tumour_size"]["median"] == 5.0
    assert result["tumour_size"]["minimum"] == 2.0
    assert result["tumour_size"]["maximum"] == 8.0
    assert result["tumour_size"]["range"] == 6.0


def test_summary_of_categorical_data_handles_all_missing_column():
    df = pd.DataFrame({
        "category": [None, None, None],
        "age": [20, 30, 40]
    })

    result = summary_of_categorical_data(df=df)
    result = result.set_index("column_name")

    assert list(result.index) == ["category"]
    assert result.loc["category", "unique_count"] == 0
    assert result.loc["category", "unique_percentage"] == 0.0
    assert result.loc["category", "missing_count"] == 3
    assert result.loc["category", "missing_percentage"] == 100.0
    assert result.loc["category", "most_frequent"] is None
    assert result.loc["category", "most_frequent_count"] == 0
    assert result.loc["category", "most_frequent_percentage"] == 0.0





def test_summary_of_numerical_data_returns_empty_dataframe_when_no_numeric_columns():
    df = pd.DataFrame({
        "city": ["London", "Leeds"],
        "passed": [True, False]
    })

    result = summary_of_numerical_data(df=df)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_summary_of_numerical_data_without_transpose():
    df = pd.DataFrame({
        "age": [10, 20, 30],
        "score": [1.0, 2.0, 3.0],
        "city": ["London", "Leeds", "Liverpool"]
    })

    result = summary_of_numerical_data(
        df=df,
        transpose=False
    )

   
    assert "mean" in result.index
    assert "median" in result.index
    assert "range" in result.index

    
    assert list(result.columns) == ["age", "score"]
    assert "city" not in result.columns

    assert result.loc["mean", "age"] == 20.0
    assert result.loc["median", "age"] == 20.0
    assert result.loc["range", "age"] == 20.0

    assert result.loc["mean", "score"] == 2.0
    assert result.loc["missing_count", "score"] == 0
    assert result.loc["unique_count", "score"] == 3


def test_summary_of_categorical_data_returns_empty_dataframe_when_no_categorical_columns():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "height": [1.7, 1.8, 1.9]
    })

    result = summary_of_categorical_data(df=df)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_summaries_of_targets_handles_all_missing_classification_target():
    df = pd.DataFrame({
        "diagnosis": [None, None, None],
        "age": [20, 30, 40]
    })

    result = summaries_of_targets(
        df=df,
        target_names="diagnosis",
        problem_types="classification"
    )

    diagnosis_summary = result["diagnosis"]

    assert diagnosis_summary["problem_type"] == "classification"
    assert diagnosis_summary["unique_count"] == 0
    assert diagnosis_summary["unique_percentage"] == 0.0
    assert diagnosis_summary["missing_count"] == 3
    assert diagnosis_summary["missing_percentage"] == 100.0
    assert diagnosis_summary["unique_values"] == []
    assert diagnosis_summary["class_counts"] == {}
    assert diagnosis_summary["class_percentages"] == {}