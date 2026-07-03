import os

def plot_render(name, plot, html_parts):
    html_parts.append(f"<h4>{name}</h4>")

    
    if isinstance(plot, dict):

        for sub_name, sub_plot in plot.items():

          
            if isinstance(sub_plot, dict):
                plot_render(sub_name, sub_plot, html_parts)

            else:
                html_parts.append(f"<h5>{sub_name}</h5>")
                html_parts.append(sub_plot.to_html(full_html=False, include_plotlyjs="cdn"))

    else:
        html_parts.append(plot.to_html(full_html=False, include_plotlyjs="cdn"))


def generate_html_report(report, report_title, output_path = "eda_report.html"):
    summary = report["summary"]
    correlations = report["correlations"]
    plots = report["plots"]

    html_parts = []


    html_parts.append(f"""

    <html>
    <head>
                        <title>{report_title} </title>
    <head/>
    <body>
                        <h1> Exploratory Data Analysis Report </h1>                      


    """)

    html_parts.append("<h2>Summary</h2>")

    for k,v in summary.items():
        html_parts.append(f"<h3> {k} </h3>")
        html_parts.append(f"<pre> {v} </pre>")




    html_parts.append("<h2> Plots </h2>")

    for area, group in plots.items():
        html_parts.append(f"<h3>{area}</h3>")
        plot_render(area, group, html_parts)


    html_parts.append("<h2> Correlations </h2>")

    html_parts.append(f"<pre>{correlations}</pre>")
    

    html_parts.append("""


    </body>
                      </html>
                      """)
    



    full_html = "\n".join(html_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return output_path



