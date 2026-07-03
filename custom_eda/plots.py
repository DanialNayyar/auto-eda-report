import plotly.express as px
import plotly.figure_factory as ff

def plot_numerical_histogram(df, column, bins = None, title = None):

  cleaned_df = df[[column]].dropna()  # double brackets = pd dataframe

  plot = px.histogram(cleaned_df, x = column, nbins = bins)

  default_title = f"Histogram of {column}"

  if title is None:
    plot.update_layout(title = default_title)
  else:
    plot.update_layout(title = title)

  plot.update_xaxes(title = column)
  plot.update_yaxes(title = "Frequency")
  plot.update_traces(marker_line_width = 1, marker_line_color = "black")

  return plot





def plot_numerical_boxplot(df, column, title = None):
  cleaned_df = df[[column]].dropna()

  plot = px.box(cleaned_df, y = column)

  default_title = f"Boxplot of {column}"

  if title is None:
    plot.update_layout(title = default_title)
  else:
    plot.update_layout(title = title)

  plot.update_yaxes(title = column)

  plot.update_traces(marker_size = 6, line_width = 1)


  return plot





def plot_categorical_count(df, column, title = None):
  cleaned_df = df[[column]].dropna()

  plot = px.histogram(cleaned_df, x = column)


  default_title = f"Count of {column}"

  if title is None:
    plot.update_layout(title = default_title)
  else:
    plot.update_layout(title = title)


  plot.update_xaxes(title = column)
  plot.update_yaxes(title = "Count")
  plot.update_traces(marker_line_width = 1, marker_line_color = "black")
  
  return plot

def plot_numerical_scatter(df, x_column, y_column, y_x_line = False, title = None ):
  cleaned_df = df[[x_column, y_column]].dropna()

  plot = px.scatter(cleaned_df, x = x_column, y = y_column)

  default_title = f"Scatter plot of {x_column} vs {y_column}"

  if title is None:
    plot.update_layout(title = default_title)
  else:
    plot.update_layout(title = title)

  plot.update_xaxes(title = x_column)
  plot.update_yaxes(title = y_column)



  if y_x_line:

    min_val = min(cleaned_df[x_column].min(), cleaned_df[y_column].min())
    max_val = max(cleaned_df[x_column].max(), cleaned_df[y_column].max())


    plot.add_shape(
        type = "line",
        x0 =min_val,
        y0 = min_val,
        x1 = max_val,
        y1 = max_val,
        line = dict(color = "red", width = 1, dash = "dash")
    )

  return plot




def plot_correlation_heatmap(df, title = None):
  num_df = df.select_dtypes(include = ["bool", "number"])

  corr_matrix = num_df.corr()


  plot = px.imshow(corr_matrix, 
                   text_auto = True,
                   zmin=-1,
                   zmax=1,
                   color_continuous_scale = "RdBu",
                   title = title)
  



  
  if title is None:
    plot.update_layout(title = "Correlation Heatmap")

  else:
    plot.update_layout(title = title)





  plot.update_xaxes(side = "top")
  
  
  return plot




def plot_numerical_kde(df, column, title = None):
  cleaned_series = df[column].dropna().values

  plot = ff.create_distplot([cleaned_series], [column], show_hist = False, show_rug = False)

  default_title = f"KDE plot of {column}"

  if title is None:
    plot.update_layout(title = default_title)
  else:
    plot.update_layout(title = title)


  
  return plot




