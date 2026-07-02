import pandas as pd
import plotly.express as px


def plot_time_series(df, time_column, value_column, add_rolling_mean = None, title = None):
  
  cleaned_df = df[[time_column, value_column]].dropna()

  cleaned_df[time_column] = pd.to_datetime(cleaned_df[time_column])

  cleaned_df = cleaned_df.sort_values(by = time_column).reset_index(drop = True)

  plot = px.line(cleaned_df, x = time_column, y = value_column)

  if add_rolling_mean is not None:
    rolling_mean = cleaned_df[value_column].rolling(window = add_rolling_mean).mean()
    plot.add_scatter(x = cleaned_df[time_column], y = rolling_mean, mode = "lines",line = dict(color = "red", width = 2) ,name = f"Rolling Mean ({add_rolling_mean})")
  
  if title is None:
    plot.update_layout(title = f"Time Series Plot of {value_column}")
  else:
    plot.update_layout(title = title)

  plot.update_xaxes(title = time_column)
  plot.update_yaxes(title = value_column)
  
  return plot