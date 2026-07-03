
def get_strong_feature_correlations(data_frame, correlation_threshold, method = "pearson"):

  """
  Finding strongly correlated features within a dataframe
  

  """

  if correlation_threshold <= 0 or correlation_threshold > 1:
    raise ValueError("Threshold must be greater than 0 and less than or equal to 1")

  if data_frame.empty:
    raise ValueError("Data frame is empty")


  numerical_df = data_frame.select_dtypes(include = ["number", "bool"])

  if numerical_df.empty:
    raise ValueError("No numerical or boolean columns in data frame")

  if numerical_df.shape[1] < 2:
    return {}

  if method is None:
    method = "pearson"

  method = method.strip().lower()

  if method not in ["pearson", "spearman", "kendall"]:
    raise ValueError("Method must be pearson, spearman, or kendall")

  final_dictionary = {}

  corr_matrix = numerical_df.corr(method = method)

  for feature in corr_matrix:


    feature_corr = corr_matrix.loc[feature].drop(feature)

    strong_features = feature_corr.abs() >= correlation_threshold

    strong_corr = feature_corr[strong_features]
    strong_sorted = strong_corr.abs().sort_values(ascending = False).index

    if not strong_sorted.empty:
      final_dictionary[feature] = feature_corr[strong_sorted].to_dict()




  return final_dictionary


def summary_of_correlations(df, correlation_threshold, method = None, return_strongest = True):
  """
  Returns a summary of the correlations between columns in a pandas dataframe.
  """

  if method is None:
    method = "pearson"

  method = method.strip().lower()

  if method not in ["pearson", "spearman", "kendall"]:
    raise ValueError("Method must be pearson, spearman, or kendall")

  if correlation_threshold <= 0 or correlation_threshold > 1:
    raise ValueError("Threshold must be greater than 0 and less than or equal to 1")
  
  numerical_df = df.select_dtypes(include = ["number", "bool"])

  if numerical_df.empty:
    raise ValueError("No numerical or boolean columns in data frame")

  if numerical_df.shape[1] < 2:
    return {}

  cols_analysed = numerical_df.columns.tolist()
  number_of_cols_analysed = numerical_df.shape[1]


  if return_strongest:
    strongest_corrs = get_strong_feature_correlations(numerical_df, correlation_threshold, method)
  else:
    strongest_corrs = None



  output_dict = {
      "correlation_method": method,
      "correlation_threshold": correlation_threshold,
      "columns_analysed": cols_analysed,
      "number_of_columns_analysed": number_of_cols_analysed,
      "correlations": numerical_df.corr(method = method).to_dict(),
      "strongest_correlations": strongest_corrs
  }


  return output_dict


