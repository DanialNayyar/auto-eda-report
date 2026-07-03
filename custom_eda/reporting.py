from .summaries import dataset_overview,summary_of_data_types,summary_of_numerical_data,summary_of_duplicates,summary_of_categorical_data


from .correlations import summary_of_correlations,get_strong_feature_correlations


from .plots import plot_categorical_count,plot_correlation_heatmap,plot_numerical_boxplot,plot_numerical_histogram,plot_numerical_kde,plot_numerical_scatter


def scatter_plot_helper(df, correlation_threshold, correlation_method):

    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    strongest_correlations = get_strong_feature_correlations(
        df[num_cols],
        method=correlation_method,
        correlation_threshold=correlation_threshold
    )

    pairs = []

    for feature, related_dict in strongest_correlations.items():
        for related_feature in related_dict.keys():
            pairs.append((feature, related_feature))

    return pairs



def generate_eda_report(df,
                        numeric_cols=None,
                        strong_corrs_for_scatter=False,
                        correlation_method="pearson",
                        correlation_threshold=0.7,
                        categorical_cols=None,
                        scatter_plot_pairs=None,
                        scatter_plot_x_y_line=True,
                        target_columns=None,
                        report_title=None):


    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    num_df = df[numeric_cols]


    overview = dataset_overview(df=df)
    data_type_summary = summary_of_data_types(df=df)
    numerical_summary = summary_of_numerical_data(df=df)
    duplicate_summary = summary_of_duplicates(df=df)
    categorical_sum = summary_of_categorical_data(df=df)

    cat_counts = {}

    for col in categorical_cols:
        cat_counts[col] = plot_categorical_count(df, column=col)

    cat_cols_plots = {
        "cat_count": cat_counts
    }

    corr_heatmap = plot_correlation_heatmap(num_df)

    corr_sum = summary_of_correlations(
        num_df,
        method=correlation_method,
        correlation_threshold=correlation_threshold
    )

    strongest_corrs = get_strong_feature_correlations(
        num_df,
        method=correlation_method,
        correlation_threshold=correlation_threshold
    )

    box_plots = {}
    num_histograms = {}
    num_kdes = {}

    for col in numeric_cols:

        box_plots[col] = plot_numerical_boxplot(df, column=col)
        num_histograms[col] = plot_numerical_histogram(df, column=col)
        num_kdes[col] = plot_numerical_kde(df, column=col)


    best_correlations = []

    if strong_corrs_for_scatter:
        best_correlations = scatter_plot_helper(df, correlation_threshold, correlation_method)
    else:
        best_correlations = scatter_plot_pairs or []

    scatter_plots = {}

    for x, y in best_correlations:
        scatter_plots[f"{x}_vs_{y}"] = plot_numerical_scatter(
            df=df,
            x_column=x,
            y_column=y,
            y_x_line=scatter_plot_x_y_line
        )


    numeric_cols_plots = {
        "box_plots": box_plots,
        "num_histograms": num_histograms,
        "num_kdes": num_kdes,
        "num_scatters": scatter_plots
    }


    all_plots = {
        "numerical_plots": numeric_cols_plots,
        "categorical_plots": cat_cols_plots,
        "correlation": {
            "correlation_heatmap":corr_heatmap}
        
    }
    summary = {
        "overview": overview,
        "data_types": data_type_summary,
        "numerical_summary": numerical_summary,
        "duplicate_summary": duplicate_summary,
        "categorical_summary": categorical_sum
    }

    correlations = {
        "correlation_summary": corr_sum,
        "strongest_correlations": strongest_corrs
    }


    return {
        "summary": summary,
        "correlations": correlations,
        "plots": all_plots
    }