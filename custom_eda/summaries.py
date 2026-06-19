import pandas as pd



def dataset_overview(df):
  """
  Returns a basic overview of a pandas dataframe.
  """

  shape = df.shape
  cols = list(df.columns)
  num_rows = df.shape[0]
  num_cols = df.shape[1]
  memory = (df.memory_usage(deep = True).sum())/(1024 ** 2)
  index_type = type(df.index).__name__

  overview = {
      "shape": shape,
      "column_names": cols,
      "number_of_rows": num_rows,
      "number_of_columns": num_cols,
      "index_type": index_type,
      "memory_usage_mb": memory
  }


  return overview



def summary_of_data_types(df):
  """
  Returns a summary of the data types in a pandas dataframe.
  """
  num_cols = []
  bool_cols = []
  date_time_cols = []
  other_cols = []
  cat_obj_col = []

  cols = {
      "numeric_cols": num_cols,
      "boolean_cols": bool_cols,
      "categorical_object": cat_obj_col,
      "datetime_col": date_time_cols,
      "other": other_cols
  }

  for col in df.columns:

    if df[col].dtype == 'bool':
      bool_cols.append(col)

    elif df[col].dtype == 'object' or df[col].dtype == 'category':
      cat_obj_col.append(col)

    elif df[col].dtype == 'int64' or df[col].dtype == 'float64':
      num_cols.append(col)

    elif df[col].dtype == 'datetime64[ns]':
      date_time_cols.append(col)

    else:
      other_cols.append(col)



  return cols






def summary_of_missing_values(df):
  """
  Returns a summary of the missing values in a pandas dataframe.
  """
  missing_count = df.isnull().sum()
  missing_count_percent = 100 * missing_count / len(df)

  missing_count_summary = pd.concat([missing_count, missing_count_percent], axis=1)

  missing_count_summary.columns = ['missing_count', 'missing_percent']
  missing_count_summary = missing_count_summary.sort_values(by='missing_count', ascending=False)

  return missing_count_summary







def summary_of_duplicates(df):
  """
  Returns a summary of the duplicate values in a pandas dataframe.
  """
  duplicate_count = int(df.duplicated().sum())
  duplicate_percent = float(100 * duplicate_count / len(df))

  duplicate_summary = {
      "duplicate_count": duplicate_count,
      "duplicate_percent": duplicate_percent
  }

  return duplicate_summary






def summary_of_numerical_data(df, transpose = True):
  """
  Returns summary stats for numerical columns in a pandas dataframe.
  """


  df_num = df.select_dtypes(include="number")

  if df_num.empty:
    return pd.DataFrame() #empty df

  else:

    overview = df_num.describe()
    overview.loc["median"] = df_num.median()
    overview.loc["range"] = df_num.max() - df_num.min()

    overview.loc['skew'] = df_num.skew()
    overview.loc['kurtosis'] = df_num.kurtosis()
    overview.loc['missing_count'] = df_num.isnull().sum()
    overview.loc['missing_percentage'] = 100 * df_num.isnull().mean()

    overview.loc['unique_count'] = df_num.nunique()
    overview.loc['unique_percentage'] = 100 * df_num.nunique() / len(df_num)



  overview = overview.round(3)
  if transpose:
    overview = overview.transpose()

  return overview






def summary_of_categorical_data(df, transpose = False):
  """
  Returns summary stats for categorical columns in a pandas dataframe.
  """
  df_cat_bool = df.select_dtypes(include =["object", "category", "bool"])
  summary = []

  if df_cat_bool.empty:
    return pd.DataFrame() #empty df

  else:
    for col in df_cat_bool.columns:
      unique_count = df_cat_bool[col].nunique()
      unique_percentage = 100 * unique_count / len(df_cat_bool)
      missing_count = df_cat_bool[col].isnull().sum()
      missing_percentage = 100 * missing_count / len(df_cat_bool)
      dtype = str(df_cat_bool[col].dtype)
      value_counts = df_cat_bool[col].value_counts()

      if value_counts.empty:
        most_frequent = None
        most_frequent_count = 0
        most_frequent_percentage = 0

      else:
        most_frequent = value_counts.index[0]
        most_frequent_count = value_counts.iloc[0]
        most_frequent_percentage = 100 * most_frequent_count / len(df_cat_bool)

      col_dict = {
        "column_name": col,
        "column_type": dtype,
        "unique_count": int(unique_count),
        "unique_percentage": float(unique_percentage),
        "missing_count": int(missing_count),
        "missing_percentage": float(missing_percentage),
        "most_frequent": most_frequent,
        "most_frequent_count": int(most_frequent_count),
        "most_frequent_percentage": float(most_frequent_percentage)
    }

      summary.append(col_dict)

    summary = pd.DataFrame(summary)


  if transpose:
    summary = summary.transpose()


  return summary





def summaries_of_targets(df, target_names = None, problem_types = None):
  
  if target_names is None:
    raise ValueError("target names is empty - needs at least one target")

  if isinstance(target_names, str):
    target_names = [target_names]

  if len(target_names) == 0:
    raise ValueError(f"Target names is empty")


  missing_targets = []

  for name in target_names:
    if name not in df.columns:
      missing_targets.append(name)
    
  if missing_targets:
    raise ValueError(f"The following are not in the dataset: {missing_targets}")
  
  
  if problem_types is None:
    raise ValueError("problem types is empty - needs at least one problem type")

  
  if isinstance(problem_types, str):
    problem_types = [problem_types]

  if len(problem_types) == 0:
    raise ValueError(f"Problem types is empty")

  if len(problem_types) != len(target_names):
    raise ValueError(f"Problem types is not the same length as target names")




  problem_types = [prob.strip().lower() for prob in problem_types]
  



  allowed_probs = ["classification", "regression"]
  missing_probs = []

  for problem in problem_types:
    if problem not in allowed_probs:
      missing_probs.append(problem)
  
  if missing_probs:
    raise ValueError(f"The following are not in the allowed problems: {missing_probs}")



  list_of_targets = {}
  for name, prob in zip(target_names, problem_types):
    data_type = str(df[name].dtype)
    unique_count = df[name].nunique()
    unique_percentage = 100 * unique_count / len(df)
    missing_count = df[name].isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    unique_values = df[name].dropna().unique().tolist()



    if prob == "classification":
      
      class_counts = df[name].value_counts()
      
      if class_counts.empty:
        class_counts_dict = {}
        class_counts_percentages = {}
      else:
        class_percentages = (100* class_counts/class_counts.sum())
        class_counts_dict = class_counts.to_dict()
        class_counts_percentages = class_percentages.to_dict()

      data_for_each_target = {
        "problem_type": prob,
        "data_type": data_type,
        "unique_count": int(unique_count),
        "unique_percentage": float(unique_percentage),
        "missing_count": int(missing_count),
        "missing_percentage": float(missing_percentage),
        "unique_values": unique_values,
                    
        "class_counts":class_counts_dict,
        "class_percentages": class_counts_percentages
      }
    
  

    elif prob == "regression":
      mean = df[name].mean()
      median = df[name].median()
      std = df[name].std()
      minimum = df[name].min()
      maximum = df[name].max()
      value_range = df[name].max() - df[name].min()
      skew = df[name].skew()
      kurtosis = df[name].kurtosis()


      data_for_each_target = {
        "problem_type": prob,
        "data_type": data_type,
        "unique_count": int(unique_count),
        "unique_percentage": float(unique_percentage),
        "missing_count": int(missing_count),
        "missing_percentage": float(missing_percentage),


        "mean": float(mean),
        "median": float(median),
        "standard_deviation": float(std),
        "minimum": float(minimum),
        "maximum": float(maximum),
        "range": float(value_range),
        "skew": float(skew),
        "kurtosis": float(kurtosis)                    

      }

   


      

    list_of_targets[name] = data_for_each_target


  return list_of_targets



