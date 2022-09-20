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
from .entities import SalienceRecord
from .utils import SDGConverter, listify


def calculate_sdg_salience(y_pred: List[Dict[int, float]]) -> SalienceRecord:
    """
    Calculate the number of texts predicted for each sdg and normalise by the largest number.

    Parameters
    ----------
    y_pred : List[Dict[int, float]]
        A list of predictions where each element is a mapping from sdg ids to predicted probabilities

    Returns
    -------
    salience : SalienceRecord
        A mapping from sdg ids to relative salience.
    """
    counter = Counter()
    for d in y_pred:
        if not isinstance(d, (dict, list, tuple)):
            continue
        counter.update(list(d))  # works with dicts, list, tuples, etc.

    # normalise
    maximum = max(counter.values()) if len(counter) > 1 else 1
    salience = SalienceRecord(dictionary={k: counter.get(k, 0) / maximum for k in range(1, 18)})
    return salience


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
