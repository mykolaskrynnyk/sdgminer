"""
This module defines plotting routines for SDG analysis.
"""
# standard library
from typing import Tuple

# data wrangling
import numpy as np

# graphs/networks
import networkx as nx

# visualisation
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from bokeh.plotting import from_networkx
from bokeh.models import (
    NodesAndLinkedEdges, Circle, HoverTool, MultiLine, Plot, Range1d, ResetTool, ColumnDataSource, LabelSet
)

# local packages
from .entities import SalienceRecord

# other settings
pio.templates.default = 'plotly_white'


def plot_sdg_salience(salience_record: SalienceRecord) -> go.Figure:
    """
    Plot a bar chart of sdg salience in the document.

    Plot an sdg-coloured barchart of relative sdg salience in the document.

    Parameters
    ----------
    salience_record : SalienceRecord
        A mapping from sdg ids to their relative salience.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A plotly bar chart figure.
    """

    fig = px.bar(
        data_frame = salience_record.df,
        x = 'sdg_id',
        y = 'salience',
        custom_data = ['sdg_name'],
        color = 'sdg_id',
        color_discrete_map = dict(salience_record.df[['sdg_id', 'sdg_color']].values)
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


def plot_sdg_salience_comparison(
        salience_record_x: SalienceRecord,
        salience_record_y: SalienceRecord,
        names: Tuple[str, str] = ('x', 'y')
    ) -> go.Figure:
    """
    Plot a grouped bar chart of two sdg salience records.

    Plot an sdg-coloured barchart of relative sdg salience in the document.

    Parameters
    ----------
    salience_record_x : SalienceRecord
        A mapping from sdg ids to their relative salience.
    salience_record_y : SalienceRecord
        A mapping from sdg ids to their relative salience.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A plotly bar chart figure.
    """
    fig = go.Figure()
    shapes = ['.', 'x']
    for name, salience_record in zip(names, [salience_record_x, salience_record_y]):
        fig.add_trace(
            go.Bar(
                name = name,
                x = salience_record.df['sdg_id'].tolist(),
                y = salience_record.df['salience'].tolist(),
                customdata = np.transpose([salience_record.df['sdg_name'].tolist()]),
                marker_color = salience_record.df['sdg_color'].tolist(),
                marker_pattern_shape = shapes.pop(),
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


def plot_sdg_graph(G: nx.Graph, show_edge_features: bool = True, width: int = 600, height: int = 800):
    """
    Plot a network graph showing linkages between sdgs.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph to be plotted.
    show_edge_features : bool, default=True
        If True, edge features, i.e. keywords, are displayed, otherwise only edge weights are shown.
    width : int, default=600
        Figure width.
    height : int, default=800
        Figure height.

    Returns
    -------
    plot : bokeh.Plot
        A Bokeh figure.
    """

    # removing nodes with tiny near-zero relative salience
    G = G.copy()
    remove_nodes = [node for node, weight in nx.get_node_attributes(G, 'weight').items() if weight < 0.01]
    G.remove_nodes_from(remove_nodes)

    plot = Plot(
        width = width,
        height = height,
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

    if show_edge_features:
        edge_tooltip = [('Weight', '@weight'), ('Linked Topics', '<br>@features{safe}')]  # {safe} does not escape HTML
    else:
        edge_tooltip = [('Weight', '@weight')]

    plot.add_tools(
        HoverTool(
            tooltips = [('SDG', '@index'), ('', '@name'), ('Salience', '@weight')],
            renderers = [graph_renderer.node_renderer]
        ),
        HoverTool(
            tooltips = edge_tooltip,
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

    # calculating edge positions
    x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
    node_labels = nx.get_node_attributes(G, 'name')  # {1: 'Goal 1...'}
    node_weights = nx.get_node_attributes(G, 'weight')  # {1: 12.345...}
    source = ColumnDataSource({
        'x': x,
        'y': y,
        'name': [str(node_name) for node_name in node_labels.keys()],
        'h_offset': [-5 if node_name < 10 else -9 for node_name in node_labels.keys()],
        # move up if a node is too small
        'v_offset': [-5 if node_weights[node_name] > 20 else 10 for node_name in node_labels.keys()],
        # make black if a node label is moved up
        'color': ['white' if node_weights[sdg] > 20 else 'black' for sdg in node_labels.keys()]
    })
    labels = LabelSet(
        x='x',
        y='y',
        text='name',
        source=source,
        x_offset='h_offset',
        y_offset='v_offset',
        text_color='color',
        # background_fill_color='black'
    )
    plot.renderers.append(labels)

    return plot
