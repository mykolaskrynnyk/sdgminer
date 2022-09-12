"""
This module defines plotting routines for SDG analysis.
"""
# standard library
from typing import Dict

# data wrangling
import numpy as np
import pandas as pd

# graphs/networks
import networkx as nx

# visualisation
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from bokeh.plotting import from_networkx
from bokeh.models import (
    EdgesAndLinkedNodes, NodesAndLinkedEdges, BoxZoomTool, Circle, HoverTool,
    MultiLine, Plot, Range1d, ResetTool, BoxSelectTool, TapTool
)

# local packages
from .utils import sdg_id2name, sdg_id2color

# other settings
pio.templates.default = 'plotly_white'


def plot_sdg_salience(sdgs: Dict[int, float]):
    """
    Plot a bra chart of sdg salience in the document.

    Plot an sdg-coloured barchart of relative sdg salience in the document.

    Parameters
    ----------
    sdgs : Dict[int, float]
        A mapping from sdg ids to their relative salience. If any sdg is missing from the mapping, it is assumed to
        have zero relative salience.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A plotly bar chart figure.
    """
    for sdg in range(1, 18):
        if sdg not in sdgs:  # fill in values for missing sdgs, if any exist
            sdgs[sdg] = 0.

    df_sdgs = pd.DataFrame(sdgs.items(), columns = ['sdg', 'salience'])
    df_sdgs.sort_values('sdg', ignore_index = True, inplace = True)
    df_sdgs['sdg_name'] = df_sdgs['sdg'].replace(sdg_id2name)
    df_sdgs['sdg'] = df_sdgs['sdg'].astype(str)  # cast to string for plotting

    fig = px.bar(
        data_frame = df_sdgs,
        x = 'sdg',
        y = 'salience',
        custom_data = ['sdg_name'],
        color = 'sdg',
        color_discrete_map = {str(k): v for k, v in sdg_id2color.items()}
    )

    fig.update_layout(
        # template = 'none',  # theme
        title = 'Relative Salience of SDGs in the Text',
        xaxis = {
            'title': 'SDG',
            'range': (-1., 17.)  # adjust the axis to fit all sdgs
        },
        yaxis = {
            'title': 'Relative Salience',
            'range': (0, 1.05)  # adjust the axis to fit all bars
        },
        showlegend = False,
        modebar_remove = ['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'lasso2d']
    )

    fig.update_traces(
        hovertemplate = '<b>%{customdata[0]}</b><br>Relative salience: %{y:.2f}'
    )

    # manually setting the background, see https://github.com/streamlit/streamlit/issues/4178
    fig.update_layout(paper_bgcolor = "white", plot_bgcolor = "white")

    return fig


def plot_sdg_salience_comparison(df_salience: pd.DataFrame):
    """
    Plot a bra chart of sdg salience in the document.

    Plot an sdg-coloured barchart of relative sdg salience in the document.

    Parameters
    ----------
    sdgs : Dict[int, float]
        A mapping from sdg ids to their relative salience. If any sdg is missing from the mapping, it is assumed to
        have zero relative salience.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A plotly bar chart figure.
    """
    fig = go.Figure()
    for idx in (1, 2):  # two columns that refer to the uploaded file names
        fig.add_trace(
            go.Bar(
                name = df_salience.columns[idx],
                x = df_salience['sdg'].tolist(),
                y = df_salience.iloc[:, idx].tolist(),
                customdata = np.transpose([df_salience['sdg_name'].tolist()]),
                marker_color = df_salience['colour'].tolist(),
                marker_pattern_shape = ('x', '.')[idx-1],
                hovertemplate = '<b>%{customdata[0]}</b><br>Relative salience: %{y:.2f}'
            )
        )

    fig.update_layout(
        # template = 'none',  # theme
        title = 'Relative Salience of SDGs in the Two Texts',
        xaxis = {
            'title': 'SDG',
            'range': (-1., 17.)  # adjust the axis to fit all sdgs
        },
        yaxis = {
            'title': 'Relative Salience',
            'range': (0, 1.05)  # adjust the axis to fit all bars
        },
        showlegend = True,
        modebar_remove = ['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'lasso2d'],
        paper_bgcolor = "white",  # manually setting the background
        plot_bgcolor = "white"  # see https://github.com/streamlit/streamlit/issues/4178
    )

    return fig


def plot_sdg_graph(G: nx.Graph):
    """
    Plot a network graph showing linkages between sdgs.
    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph to be plotted.

    Returns
    -------
    plot : bokeh.Plot
        A Bokeh figure.
    """
    plot = Plot(
        # width = 800,
        height = 800,
        x_range = Range1d(-1.15, 1.15),
        y_range = Range1d(-1.15, 1.15),
        title = 'SDG Connections Graph'
    )

    graph_renderer = from_networkx(
        graph = G,
        layout_function = nx.shell_layout,  # alternative layouts might be preferred, e.g., nx.kamada_kawai_layout,
        scale = 1,
        center = (0,0)
    )

    plot.add_tools(
        HoverTool(
            tooltips = [('SDG', '@index'), ('', '@name'), ('Salience', '@weight')],
            renderers = [graph_renderer.node_renderer]
        ),
        HoverTool(
            tooltips = [('Weight', '@weight'), ('Linked Topics', '<br>@features{safe}')],  # {safe} does not escape HTML
            renderers = [graph_renderer.edge_renderer]
        ),
        # BoxZoomTool(),
        ResetTool(),
        # TapTool(),
        # BoxSelectTool()
    )

    # configuring nodes
    graph_renderer.node_renderer.glyph = Circle(size = 'weight', fill_color = 'color')
    #graph_renderer.node_renderer.hover_glyph = Circle(fill_color= 'yellow')
    #graph_renderer.node_renderer.selection_glyph = Circle(fill_color = 'red')

    # configuring edges
    graph_renderer.edge_renderer.glyph = MultiLine(line_color = '#808080', line_alpha = .9, line_width = 'weight')
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color = 'red', line_width = 'weight') # line_width=5
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color = 'pink', line_width = 'weight')

    # configuring policies
    graph_renderer.selection_policy = NodesAndLinkedEdges()
    #graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)
    return plot
