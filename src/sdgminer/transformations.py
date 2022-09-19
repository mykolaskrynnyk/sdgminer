"""
Transformation functions for reshaping data and creating derived data structures.
"""
# standard library
import re
from itertools import combinations
from collections import Counter
from typing import List, Dict, Tuple, Union

# data wrangling
import numpy as np
import pandas as pd

# graphs/networks
import networkx as nx

# local packages
from .utils import sdg_id2name, sdg_id2color, listify


def calculate_sdg_salience(y_pred: List[Dict[int, float]]) -> Dict[int, float]:
    """
    Calculate the number of texts predicted for each sdg and normalise by the most largest number.

    Parameters
    ----------
    y_pred : List[Dict[int, float]]
        A list of predictions where each element is a mapping from sdg ids to predicted probabilities

    Returns
    -------
    sdg2salience : Dict[int, float]
        A mapping from sdg ids to relative salience.
    """
    counter = Counter()
    for d in y_pred:
        if not isinstance(d, (dict, list, tuple)):
            continue
        counter.update(list(d))  # works with dicts, list, tuples, etc.

    # normalise
    maximum = max(counter.values()) if len(counter) > 1 else 1
    sdg2salience = {k: counter.get(k, 0) / maximum for k in range(1, 18)}
    return sdg2salience


def get_sdg_features(df_texts: pd.DataFrame, sdg2features: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """
    Get most common features per SDG using pre-defined SDG features.

    Calculates most common features (tokens/phrases) for each group of SDG texts. In so doing, the function considers
    only features from a pre-defined list of terms for each SDG. It only uses the predictions from a multilabel model.

    Parameters
    ----------
    df_texts : pd.DataFrame
        A dataframe of containing a 'text_' column and columns with predicted sdgs.
    sdg2features : Dict[int, List[str]]
        A mapping from integer numbers (sdg ids) to a list of strings that define features relevant for to this sdg.

    Returns
    -------
    sdg_id2top_features : Dict[int, List[str]]
        A mapping from integer numbers (sdg ids) to a list of strings that define top features sorted in descending
        order of importance. Does not include any sdg for which no text was assigned.
    """
    assert 'sdgs_multilabel' in df_texts.columns, '"sdgs_multilabel" column in missing'

    df_sdgs = df_texts.dropna(subset = ['sdgs_multilabel']) \
        .explode('sdgs_multilabel') \
        .groupby('sdgs_multilabel') \
        .agg({'text_': lambda x: ' '.join(list(x))})

    sdg_id2top_features = dict()
    for sdg in df_sdgs.index:
        # top predictors from a multiclass logistic regression model, excludes uni-grams
        features = [feature for feature in sdg2features[sdg] if ' ' in feature]
        text = df_sdgs.loc[sdg, 'text_']
        feature2frequency = {feature: text.count(feature) for feature in features}
        top_features = [
            feature for feature, frequency in sorted(feature2frequency.items(), key = lambda x: x[1], reverse = True)
            if frequency > 0  # exclude zero frequency features
        ]
        sdg_id2top_features[sdg] = top_features  # may be an empty list
    return sdg_id2top_features


def get_sdg_features_pairwise(df_texts: pd.DataFrame, sdg2features: Dict[int, List[str]]) -> Dict[Tuple[int], List[str]]:
    """
    Get most common features for all pairs of sdgs that co-occur.

    Calculates most common features for each pair of sdgs from texts that are assigned to more than one sdg. In theory,
    this is expected to correspond to mixed-sdg topics, i.e., "sdg synergies". The function considers only features
    from a pre-defined list of terms for each SDG. It only uses the predictions from a multilabel model.

    Parameters
    ----------
    df_texts : pd.DataFrame
        A dataframe of containing a 'text_' column and columns with predicted sdgs.
    sdg2features : Dict[int, List[str]]
        A mapping from integer numbers (sdg ids) to a list of strings that define features relevant for to this sdg.

    Returns
    -------
    sdg_ids2top_features: Dict[Tuple[str], List[str])
        A mapping from each pair of co-occuring SDGs to a list of strings that define top features sorted in descending
        order of importance. Does not include any sdg pair for which no text was assigned.
    """
    # subset only those texts that have more than 1 sdg label predicted
    assert 'sdgs_multilabel' in df_texts.columns, '"sdgs_multilabel" column in missing'

    df_texts = df_texts.loc[df_texts['sdgs_multilabel'].str.len().ge(2)].copy()
    df_texts['sdgs'] = df_texts['sdgs_multilabel'] \
        .apply(list) \
        .apply(sorted) \
        .apply(lambda x: list(combinations(x, 2)))  # consider all pairwise combinations
    df_sdgs = df_texts.explode('sdgs') \
        .groupby('sdgs', as_index = False) \
        .agg({'text_': lambda x: ' '.join(list(x))})  # all pairwise combiations
    df_sdgs['sdgs'] = df_sdgs['sdgs'].apply(tuple)  # each cell is a tuple of two sdgs

    sdg_ids2top_features = dict()
    for idx, row in df_sdgs.iterrows():

        # combine features from both sdgs
        features = sdg2features[row['sdgs'][0]] + sdg2features[row['sdgs'][1]]
        features = [feature for feature in features if ' ' in feature]
        feature2frequency = {feature: row['text_'].count(feature) for feature in features}
        top_features = [
            feature for feature, frequency in sorted(feature2frequency.items(), key = lambda x: x[1], reverse = True)
            if frequency > 0  # exclude zero frequency features
        ]
        sdg_ids2top_features[row['sdgs']] = top_features[:15]  # keep only top 15 features for each pair
    return sdg_ids2top_features


def construct_sdg_graph(df_texts: pd.DataFrame, add_edge_features: bool = False, sdg2features: Dict[int, List[str]] = None) -> nx.Graph:
    """
    Construct a graph of relationships between SDGs in the texts.

    Creates an nx.Graph using pairwise features from co-occurence of SDGs. The output of this function is used for
    plotting.

    Parameters
    ----------
    df_texts : pd.DataFrame
        A dataframe of containing a 'text_' column and columns with predicted sdgs.
    add_edge_features : bool, default=False
        A flag to indicate whether co-occurrence keywords should be added as edge attributes.
    sdg2features : Dict[int, List[str]]
        A mapping from integer numbers (sdg ids) to a list of strings that define features relevant for to this sdg.

    Returns
    -------
    G : nx.Graph
        A graph of interconnections between SDGs that is used for plotting.
    """
    assert 'sdgs_multilabel' in df_texts.columns, '"sdgs_multilabel" column in missing'

    X = np.zeros((17, 17))  # adjacency matrix with occurrence counts on the diagonal
    for sdg_ids2probs in df_texts['sdgs_multilabel'].dropna().values:

        # loop through each pairwise combination and count co-occurrence
        for i, j in combinations(sdg_ids2probs.keys(), 2):
            X[i-1, j-1] += 1
            X[j-1, i-1] += 1

        # add occurrence counts on the diagonal
        for k in sdg_ids2probs.keys():
            X[k-1, k-1] += 1 # salience of an SDG, i.e., occurrence count
    assert (X == X.T).all(), 'X is an adjacency matrix, it must be symmetric.'

    df_sdgs = pd.DataFrame(X, columns = range(1, 18), index = range(1, 18))
    df_sdgs = df_sdgs.reset_index() \
        .rename({'index': 'from'}, axis = 1) \
        .melt(id_vars = 'from', var_name = 'to', value_name = 'weight')

    # get unique pair ids to remove duplicates
    df_sdgs['id'] = df_sdgs[['from', 'to']].astype(str).apply(list, axis = 1).apply(sorted).str.join('-')
    df_sdgs.drop_duplicates(subset = 'id', inplace = True)
    df_sdgs = df_sdgs.query('`from` != `to`').copy()  # removing edges to itself

    # sanity check: total number of pairs minus the number of duplicated pairs minus self-edges to itself
    assert df_sdgs.shape[0] == 17 * 17 - (17 * 17 - 17) / 2 - 17

    # normalising the weight to the range from 0 to 10
    df_sdgs['weight'] = df_sdgs['weight'].divide(df_sdgs['weight'].max()).multiply(10)
    df_sdgs = df_sdgs.query(
        expr = 'weight > @threshold and weight >= 5',
        local_dict = {'threshold': df_sdgs['weight'].quantile(.5)}  # only keep edges whose weights are above the median
    )

    # constructing the graph
    G = nx.Graph()
    nodes = list(range(1, 18))  # always add all nodes
    G.add_nodes_from(nodes)
    edges = df_sdgs[['from', 'to', 'weight']].apply(tuple, axis = 1).tolist()
    G.add_weighted_edges_from(edges)

    # add edge attributes, if any are given
    if add_edge_features:
        sdgs2features = get_sdg_features_pairwise(df_texts, sdg2features)
        edge_attributes = {k: {'features': listify(v)} for k, v in sdgs2features.items()}
        nx.set_edge_attributes(G, edge_attributes)

    # add node attributes
    weight_max = np.diag(X).max()
    for node in G.nodes:
        G.nodes[node]['name'] = sdg_id2name[node]
        # nodes are from 1 to 17, indices are from 0 to 16
        G.nodes[node]['weight'] = X[node-1, node-1] / weight_max * 100  # scale the node weight
        G.nodes[node]['color'] = sdg_id2color[node]

    return G


# TODO: improve the function to account for the fact that features are derived after preprocessing
# TODO: accommodate multi-label settings by accepting a dict of Dict[feature, tags], e.g. {'poverty': [1, 2]}
# TODO: fix incorrect multi-word highlights
def naively_match(text: str, features: List[str], tag: str = 'MATCH', color: str = '#34568B') -> List[Union[str, Tuple[str]]]:
    """
    Use naive matching to annotate a text with coloured spans. The output is used in the annotated_text function to
    display colour-coded texts to the user.

    Given a list of features, naively matches them to a text and highlights the matching spans using a colour and tag.
    The output is used in the annotated_text function to display colour-coded texts to the user in a streamlit app.
    The default color "#34568B" is classic blue.

    Parameters
    ----------
    text : str
        An input text to annotate.
    features : List[str]
        A list of string to match in the text.
    tag : str
        A tag to be displayed with the matching span.
    color : str
        A hex colour to use to highlight spans.

    Returns
    -------
    annotated_text : List[Union[str, Tuple[str]]]
        A list where an element can be a tuple if it is a matched span of text or a string if it is an unmatched text.
        For example, ["This is not matched", ("but this is", "MATCH", "#34568B")]

    Notes
    -----
    This is a naive implementation that does not account for the fact that features are derived after preprocessing
    the text. It is also not the most efficient implementation.
    """
    spans_all = list()
    matched_features = list()
    features.sort(key = len, reverse = True)  # from longest to shortest

    for feature in features:
        # if a feature is a substring of one of the matched features, skip it
        if any(feature in feature_ for feature_ in matched_features):
            continue
        spans = [match.span() for match in re.finditer(rf'\b{feature}\b', text)]
        if not spans:
            continue
        spans_all.extend(spans)
        matched_features.append(feature)

    if not spans_all:
        return text
    spans_all.sort(key = lambda x: x[0])  # converting to the sequential ordering by index

    annotated_text = list()
    offset = 0
    for start, end in spans_all:
        chunk = text[offset: start]  # unmatched text span
        match = text[start: end]  # matched text span
        annotated_text.extend([chunk, (match, tag, color)])
        offset = end  # move offset to include the next chunk
    annotated_text.append(text[end:])  # add the remaining part of the tex, if any
    return annotated_text
