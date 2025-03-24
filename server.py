from flask import Flask, render_template, request, Response
import pandas as pd
from query import AskCSO
import json
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.colors as pc

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    search_query = request.args.get('query', '')

    def event_stream(query):
        gen = AskCSO(query)
        final_data_sent = False

        try:
            while True:
                item = next(gen)
                # Skip any final status indicator messages.
                if isinstance(item, str) and item.startswith("\nFinal Results:"):
                    continue

                # When we encounter the DataFrame, process final outputs.
                if isinstance(item, pd.DataFrame):
                    df = item
                    try:
                        # The next three yields are llm_title, supertitle, and table_title.
                        llm_title = next(gen)      # llm_title
                        suptitle_text = next(gen)  # supertitle
                        dataset_title = next(gen)  # table_title
                    except StopIteration:
                        yield "event: error\ndata: Incomplete final data package\n\n"
                        return

                    # Set the global variables required by our plotting functions.
                    globals()['df'] = df
                    globals()['llm_title'] = llm_title

                    # Instead of opening the graph in a new window, get the HTML div.
                    graph_div = plot_graph()

                    # Prepare final payload including the graph HTML div.
                    payload = {
                        "graph_div": graph_div,
                        "title_text": llm_title,
                        "suptitle_text": suptitle_text,
                        "dataset_title": dataset_title
                    }

                    yield f"event: final\ndata: {json.dumps(payload)}\n\n"
                    final_data_sent = True
                    break
                else:
                    # Send progress messages.
                    yield f"data: {json.dumps({'message': item})}\n\n"
        except StopIteration:
            pass
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

        if not final_data_sent:
            yield "event: error\ndata: No data found\n\n"

    return Response(event_stream(search_query), mimetype="text/event-stream")

def plot_graph():
    """
    Chooses the appropriate plotting function based on the structure of the DataFrame.
    Relies on globals 'df' and 'llm_title' being set.
    Returns:
        An HTML div (string) with the plot.
    """
    len_indexes = len(df.index)
    num_indexes = df.index.nlevels
    num_cols = len(df.columns)
    print("num_cols", num_cols,  "num_indexes", num_indexes)
    
    if num_cols == num_indexes == len_indexes == 1:
        print("plotting stat")
        return plot_stat()
    elif num_indexes == 1:
        print("plotting plot_one_level_index()")
        return plot_one_level_index()
    elif num_cols == 1 and num_indexes == 2:
        print("running plot_two_level_index")
        return plot_two_level_index()
    else:
        print("attempting to plot two level index anyway")
        return plot_two_level_index()

def plot_stat():
    """Plot a single numeric statistic using Plotly Indicator and return an HTML div."""
    fig = go.Figure(go.Indicator(
        mode="number",
        value=df.iloc[0, 0],
        title={"text": llm_title}
    ))
    return pyo.plot(fig, include_plotlyjs='cdn', output_type='div', config={"displayModeBar": False})

def plot_one_level_index():
    """Plot a DataFrame with a single level index and return an HTML div."""
    fig = go.Figure()
    
    # Single column: create a bar plot with a gradient along the x-axis.
    if len(df.columns) == 1:
        col = df.columns[0]
        n = len(df)
        # Compute a gradient color for each bar using the "Emrld" colorscale.
        colors = [pc.sample_colorscale("Emrld", i/(n-1))[0] for i in range(n)]
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[col],
            marker=dict(color=colors),
            name=col,
            opacity=0.85
        ))
        showlegend = False
        y_title = col  # Use the column name for the y-axis label.
    else:
        # Multiple columns: create line plots with legend.
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                opacity=0.85
            ))
        showlegend = True
        y_title = "Value"
    
    # Determine x-axis label based on the DataFrame's index name(s).
    if isinstance(df.index, pd.MultiIndex):
        x_label = " | ".join([str(name) for name in df.index.names if name])
    else:
        x_label = df.index.name if df.index.name is not None else ""
    
    # Update layout with title and legend settings.
    fig.update_layout(
        title={'text': llm_title, 'x': 0.5, 'xanchor': 'center'},
        showlegend=showlegend,
    )
    
    # Update x-axis settings.
    fig.update_xaxes(
        tickangle=45,
        nticks=20,
        fixedrange=True,
        title_text=x_label
    )
    
    # Update y-axis settings.
    fig.update_yaxes(
        tickformat=",.0f",
        type="linear",
        fixedrange=True,
        title_text=y_title
    )
    
    return pyo.plot(fig, include_plotlyjs='cdn', output_type='div', config={"displayModeBar": False})

def plot_two_level_index():
    """
    Plot a DataFrame with a two-level MultiIndex and return an HTML div.
    Assumes the DataFrame has a two-level MultiIndex.
    """
    # Ensure we have a two-level index.
    if df.index.nlevels != 2:
        raise ValueError("DataFrame must have a two-level MultiIndex.")
    
    # Determine which index level has fewer unique values.
    unique_counts = [df.index.get_level_values(i).nunique() for i in range(2)]
    if unique_counts[0] <= unique_counts[1]:
        group_level = 0
        x_level = 1
    else:
        group_level = 1
        x_level = 0
    
    # Get unique groups.
    groups = df.index.get_level_values(group_level).unique()
    
    fig = go.Figure()
    
    # Choose the value column to plot (here, the first column).
    value_col = df.columns[0]
    
    # Create a trace for each group.
    for group in groups:
        # Subset the DataFrame for the current group.
        sub_df = df[df.index.get_level_values(group_level) == group]
        # Sort the subset by the x-axis level.
        sub_df = sub_df.sort_index(level=x_level)
        # Get x-values from the other index level.
        x_values = sub_df.index.get_level_values(x_level)
        # Get y-values from the chosen data column.
        y_values = sub_df[value_col]
        
        # Create line or bar plot based on x-values type.
        if pd.api.types.is_numeric_dtype(x_values):
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                name=str(group),
                opacity=0.85
            ))
        else:
            fig.add_trace(go.Bar(
                x=x_values,
                y=y_values,
                name=str(group),
                opacity=0.85
            ))
    
    # Determine the x-axis label based on the name of the x_level.
    x_label = df.index.names[x_level] if df.index.names[x_level] is not None else ""
    
    # Update layout.
    fig.update_layout(
        title={'text': llm_title, 'x': 0.5, 'xanchor': 'center'},
        showlegend=True,
    )
    fig.update_xaxes(
        title_text=x_label,
        tickangle=45,
        fixedrange=True
    )
    fig.update_yaxes(
        tickformat=",.0f",
        fixedrange=True
    )
    
    return pyo.plot(fig, include_plotlyjs='cdn', output_type='div', config={"displayModeBar": False})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
    #app.run(debug=False)
