"""
This module defines Manager classes which contain high-level abstractions for interacting with the package.
"""

# standard library
import warnings
from typing import Tuple, Dict
from collections import deque, Counter
from itertools import combinations

# data wrangling
import numpy as np
import pandas as pd

# graphs/networks
import networkx as nx

# local packages
from .extraction import TextExtractor
from .preprocessing import TextCleaner
from .transformations import calculate_sdg_salience
from .models import SDGModels
from .entities import SalienceRecord
from .plotting import plot_sdg_salience, plot_sdg_graph
from .utils import SDGConverter, listify


class DocumentManager(object):
    def __init__(self, model_dir: str, model_version: str):
        self.__filepath = None
        self.__text_extractor = TextExtractor()
        self.__text_cleaner = TextCleaner()
        self.__models = SDGModels(path=model_dir, version=model_version)
        self.__converter = SDGConverter()
        self.__sentences = None
        self.__paragraph_len = 1
        self.__df_texts = None
        self.__allowed_models = {'multiclass', 'multilabel'}
        self.__sdg_relevant = None

    def __repr__(self):
        return 'DocumentManager()'

    def __str__(self):
        return 'DocumentManager()'

    def extract_from_text(self, text: str) -> str:
        self.__sentences = list(self.__text_cleaner.sentencise(text=text))

    def extract_from_file(self, filepath: str) -> str:
        self.__text_extractor.extract_text(filepath=filepath)
        text = self.__text_extractor.text
        self.__filepath = filepath
        self.extract_from_text(text=text)

    def extract_from_url(self, url):
        raise NotImplementedError

    def __aggregate(self, paragraph_len : int = 3) -> pd.DataFrame:
        """
        Assemble longer texts from sentences.

        This is used to create variable-length text structures from sentence-level observations. Uses window sliding
        with the step sice of 1.

        Parameters
        ----------
        paragraph_len : int
            An integer number indicating how many sentences to include in a paragraph.

        Returns
        -------
        df_paragraphs : pd.DataFrame
            A dataframe of paragraph-level data.
        """
        text_records = list()
        text_id = deque()
        text_block = deque()
        text_block_ = deque()

        # sliding window with a step size of 1
        for idx, (sent, sent_) in enumerate(self.__sentences):
            text_id.append(str(idx))
            text_block.append(sent)
            text_block_.append(sent_)
            if len(text_block) == paragraph_len:
                text_record = ('-'.join(text_id), ' '.join(text_block), ' '.join(text_block_))
                text_records.append(text_record)
                text_id.popleft()
                text_block.popleft()
                text_block_.popleft()

        return text_records

    def analyse(self, granularity: str):
        if granularity == 'sentence':
            df_texts = pd.DataFrame(
                data=self.sentences,
                index=map(str, range(len(self.sentences))),
                columns=['text', 'text_']
            )
        elif granularity == 'paragraph':
            df_texts = pd.DataFrame(
                data=self.__aggregate(paragraph_len=6),
                columns=['text_id', 'text', 'text_']
            )
            df_texts.set_index('text_id', inplace=True)
        else:
            raise ValueError('`granularity` must be either "sentence" or "paragraph"')

        df_preds = self.__models.predict(corpus=df_texts['text_'].values, ids=df_texts.index, threshold=.5)
        self.__df_texts = df_texts.join(df_preds, how='left')
        return self

    def get_salience(self, model_type: str) -> SalienceRecord:
        assert model_type in self.__allowed_models
        self.check_relevance()
        salience_record = calculate_sdg_salience(y_pred=self.__df_texts[model_type].tolist())
        salience_record.mode_type = model_type
        return salience_record

    def plot_salience(self, model_type: str):
        self.check_relevance()
        salience_record = self.get_salience(model_type=model_type)
        fig = plot_sdg_salience(salience_record=salience_record)
        return fig

    def get_top_features(self, model_type: str, max_features: int = 100, exclude_unigrams: bool = True):
        """
        Get most common features per SDG using pre-defined SDG features.

        Calculates most common features (tokens/phrases) for each group of SDG texts. In so doing, the function considers
        only features from a pre-defined list of terms for each SDG.

        Parameters
        ----------
        model_type : {'multiclass', 'multilabel'}
            A name of the model type to be used.
        max_features : int
            The maximum number of features per sdg to return.
        exclude_unigrams : bool
            If set to True, only ngrams are returned, otherwise unigrams are included too.

        Returns
        -------
        top_features : Dict[int, List[str]]
            A mapping from integer numbers (sdg ids) to a list of strings that define top features sorted in descending
            order of importance.
        """
        assert model_type in self.__allowed_models
        if not self.check_relevance():
            return None

        sdg2features = self.__models.get_clean_features()
        top_features = dict()

        for sdg in range(1, 18):
            if exclude_unigrams:
                features = [feature for feature in sdg2features[sdg] if ' ' in feature]
            else:
                features = sdg2features[sdg]
            counter = Counter()
            mask = self.__df_texts[model_type].str.get(sdg).notna()
            for text in self.__df_texts.loc[mask, 'text_'].values:
                feature2count = {feature: text.count(feature) for feature in features}
                counter.update(feature2count)
            top_features[sdg] = [feature for feature, count in counter.most_common() if count > 0][:max_features]

        return top_features

    @property
    def sentences(self):
        return self.__sentences

    @property
    def df_texts(self):
        return self.__df_texts

    def get_top_features_pairwise(self, max_features: int = 15, exclude_unigrams: bool = True):
        """
        Get most common features for all pairs of sdgs that co-occur.

        Calculates most common features for each pair of sdgs from texts that are assigned to more than one sdg. In theory,
        this is expected to correspond to mixed-sdg topics, i.e., "sdg synergies". The function considers only features
        from a pre-defined list of terms for each SDG. It only uses the predictions from a multilabel model.

        Parameters
        ----------
        max_features : int
            The maximum number of features per sdg to return.
        exclude_unigrams : bool
            If set to True, only ngrams are returned, otherwise unigrams are included too.

        Returns
        -------
        top_features: Dict[Tuple[str], List[str])
            A mapping from each pair of co-occuring SDGs to a list of strings that define top features sorted in descending
            order of importance. Does not include any sdg pair for which no text was assigned.
        """
        if not self.check_relevance():
            return None
        sdg_pairs = list(combinations(range(1, 18), 2))
        sdg2features = self.__models.get_clean_features()
        df_texts = self.__df_texts
        top_features = dict()

        for pair in sdg_pairs:

            features = sdg2features[pair[0]] + sdg2features[pair[1]]
            if exclude_unigrams:
                features = [feature for feature in features if ' ' in feature]
            mask = df_texts['multilabel'].str.get(pair[0]).notna() & df_texts['multilabel'].str.get(pair[1]).notna()
            if mask.sum() == 0:
                continue
            counter = Counter()
            for text in df_texts.loc[mask, 'text_'].values:
                feature2count = {feature: text.count(feature) for feature in features}
                counter.update(feature2count)
            top_features[pair] = [feature for feature, count in counter.most_common() if count > 0][:max_features]
        return top_features

    def construct_sdg_graph(self) -> nx.Graph:
        """
        Construct a graph of relationships between SDGs in the texts.

        Creates an nx.Graph using pairwise features from co-occurence of SDGs. The output of this function is used for
        plotting.

        Returns
        -------
        G : nx.Graph
            A graph of interconnections between SDGs that is used for plotting.
        """
        if not self.check_relevance():
            return None
        X = np.zeros((17, 17))  # adjacency matrix with occurrence counts on the diagonal
        for sdg_ids2probs in self.__df_texts['multilabel'].dropna().values:

            # loop through each pairwise combination and count co-occurrence
            for i, j in combinations(sdg_ids2probs.keys(), 2):
                X[i - 1, j - 1] += 1
                X[j - 1, i - 1] += 1

            # add occurrence counts on the diagonal
            for k in sdg_ids2probs.keys():
                X[k - 1, k - 1] += 1  # salience of an SDG, i.e., occurrence count
        assert (X == X.T).all(), 'X is an adjacency matrix, it must be symmetric.'

        df_sdgs = pd.DataFrame(X, columns=range(1, 18), index=range(1, 18))
        df_sdgs.reset_index(inplace=True)
        df_sdgs.rename({'index': 'from'}, axis=1, inplace=True)
        df_sdgs = df_sdgs.melt(id_vars='from', var_name='to', value_name='weight')

        # get unique pair ids to remove duplicates
        df_sdgs['id'] = df_sdgs[['from', 'to']].astype(str).apply(list, axis=1).apply(sorted).str.join('-')
        df_sdgs.drop_duplicates(subset='id', inplace=True)
        df_sdgs = df_sdgs.query('`from` != `to`').copy()  # removing edges to itself

        # sanity check: total number of pairs minus the number of duplicated pairs minus self-edges to itself
        assert df_sdgs.shape[0] == 17 * 17 - (17 * 17 - 17) / 2 - 17

        # normalising the weight to the range from 0 to 10
        df_sdgs['weight'] = df_sdgs['weight'].divide(df_sdgs['weight'].max()).multiply(10)
        df_sdgs = df_sdgs.query(
            expr='weight > @threshold and weight >= 5',
            local_dict={'threshold': df_sdgs['weight'].quantile(.5)}  # keep edges whose weights are above the median
        )

        # constructing the graph
        G = nx.Graph()
        nodes = list(range(1, 18))  # always add all nodes
        G.add_nodes_from(nodes)
        edges = df_sdgs[['from', 'to', 'weight']].apply(tuple, axis=1).tolist()
        G.add_weighted_edges_from(edges)

        # add edge attributes, i.e., keywords
        sdgs2features = self.get_top_features_pairwise(max_features=15, exclude_unigrams=True)
        edge_attributes = {k: {'features': listify(v)} for k, v in sdgs2features.items()}
        nx.set_edge_attributes(G, edge_attributes)

        # add node attributes
        weight_max = np.diag(X).max()
        for node in G.nodes:
            G.nodes[node]['name'] = self.__converter.id2name(node)
            # nodes are from 1 to 17, indices are from 0 to 16
            G.nodes[node]['weight'] = X[node - 1, node - 1] / weight_max * 100  # scale the node weight
            G.nodes[node]['color'] = self.__converter.id2color(node)

        return G

    def plot_sdg_graph(self, show_edge_features: bool = True):
        if not self.check_relevance():
            return None
        G = self.construct_sdg_graph()
        fig = plot_sdg_graph(G, show_edge_features=show_edge_features)
        return fig

    def check_relevance(self) -> bool:
        relevant_multiclass = self.__df_texts['multiclass'].notna().any()
        relevant_multilabel = self.__df_texts['multilabel'].notna().any()
        if not relevant_multiclass and not relevant_multilabel:
            warnings.warn(f'This document seems to have no SDG-related content.')
            return False
        elif not relevant_multiclass:
            warnings.warn(f'This document seems to have no SDG-related content as per multiclass model.')
            return False
        elif not relevant_multilabel:
            warnings.warn(f'This document seems to have no SDG-related content as per multilabel model.')
            return False
        else:
            return True