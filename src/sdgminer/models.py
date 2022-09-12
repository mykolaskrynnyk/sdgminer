"""
This module defines SDGModels class that is used as an interface to interact with the trained machine learning models.
"""
# standard library
import os
from typing import List, Dict, Any
from collections import defaultdict

# data wrangling
import numpy as np
import pandas as pd

# utils
import joblib


class SDGModels(object):
    """
    Load pre-trained models and make predictions on texts.

    This is a container-class to interact with pre-trained ML models..

    Parameters
    ----------
    path : List[str]
        A path to the directory where models are stored.
    version : str
        A string version of the models to use.
    """
    def __init__(self, path: List[str], version: str):
        self.__models = {
            'binary': joblib.load(os.path.join(*path, f'clf-logreg-binary-v{version}.joblib')),
            'multiclass': joblib.load(os.path.join(*path, f'clf-logreg-multiclass-v{version}.joblib')),
            'multilabel': joblib.load(os.path.join(*path, f'clf-mlp-multilabel-v{version}.joblib')),
        }

    def __repr__(self) -> str:
        return 'SDGModels()'

    def __str__(self) -> str:
        return str(self.__models)

    def __call__(self, corpus: List[str], threshold: float = .5) -> pd.DataFrame:
        """
        Predict sdg labels for a given corpus.

        This is a wrapper around SDGModels.predict() to make predictions easier.

        Parameters
        ----------
        corpus : List[str]
            A list of texts to predict sdgs for.
        threshold: float, default=.5
            A minimum probability threshold to assign an sdg label.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe each column corresponding to each model's predictions. Indices are aligned with the corpus.
        """
        df_probs = self.predict(corpus = corpus, threshold = threshold)
        return df_probs

    def predict_relevant(self, corpus: List[str], ids: List[Any]) -> pd.DataFrame:
        """
        Predict if texts in the corpus are related to sdgs at all.

        Uses a binary classifier trained on sdg-relevant and sdg-irrelevant texts. Provides a probability estimate
        that a given text is related to sdgs.

        Parameters
        ----------
        corpus : List[str]
            A list of texts to predict sdgs for.
        ids : List[Any], optional
            An list of ids to be used as indices in the output dataframe.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe with 'prob_relevant' column indicating the probability a text is relevant. Indices are taken
            from ids argument.
        """
        df_probs = pd.DataFrame({'prob_relevant': self.__models['binary'].predict_proba(corpus)[:, 1]}, index = ids)
        df_probs.index.name = 'text_id'
        return df_probs

    def predict_sdgs(self, model_type: str, corpus: List[str], ids: List[Any] = None, threshold: float = .5) -> pd.DataFrame:
        """
        Predict sdg labels for texts in the corpus using either a multiclass or multilabel model.

        Uses one of the two pre-trained models to assign sdg labels to texts in the corpus. For each text provides a
        dict of predicted sdgs and their respective probabilities. Excludes any predictions that have a probability
        lower that the threshold.

        Parameters
        ----------
        corpus : List[str]
            A list of texts to predict sdgs for.
        ids : List[Any], optional
            A list of ids to be used as indices in the output dataframe.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe with one column mapping sdgs to predicted probabilities for each text. Indices are taken
            from ids argument.
        """
        assert model_type == 'multiclass' or model_type == 'multilabel'
        assert 0 < threshold < 1, 'theshold must be in the range (0, 1).'
        if ids is None:
            ids = range(len(corpus))
        y_probs = self.__models[model_type].predict_proba(corpus)
        df_probs = pd.DataFrame(y_probs, columns = list(range(1, 18)))
        df_probs.insert(0, 'text_id', ids)
        df_probs = df_probs.melt(id_vars = 'text_id', var_name = 'sdg', value_name = 'prob')\
            .query('prob > @threshold').sort_values('prob', ascending = False)

        col_name = f'sdgs_{model_type}'
        if df_probs.empty:
            df_probs = df_probs.set_index('text_id').reindex([col_name], axis = 1)
        else:
            df_probs = df_probs.groupby('text_id')\
                .apply(lambda row: dict(zip(row['sdg'], row['prob'].round(2))))\
                .to_frame(name = col_name)
        return df_probs

    def predict(self, corpus: List[str], ids: List[Any] = None, threshold: float = .5) -> pd.DataFrame:
        """
        Predict sdg labels for a given corpus.

        Uses each model to predict sdgs and store them in a column. If ids are passed, these are used as indices in
        the output dataframe. If not, a range index is used.

        Parameters
        ----------
        corpus : List[str]
            A list of texts to predict sdgs for.
        ids : List[Any], optional
            An optional list of ids to be used as indices in the output dataframe.
        threshold: float, default=.5
            A minimum probability threshold to assign an sdg label.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe each column corresponding to each model's predictions. Indices are aligned with the corpus.
        """
        assert len(corpus) == len(ids), 'corpus and ids must be of the same length.'
        df_probs = self.predict_relevant(corpus = corpus, ids = ids)
        for model_type in ('multiclass', 'multilabel'):
            df_temp = self.predict_sdgs(model_type = model_type, corpus = corpus, ids = ids, threshold = threshold)
            df_probs = df_probs.join(df_temp, how = 'left')
        return df_probs

    @property
    def features(self) -> Dict[int, List[str]]:
        """
        Extract features that are top predictions for each SDG from the multiclass classifier.

        Returns
        -------
        sdg2features : Dict[int, List[str]]
            A mapping from integer numbers (sdg ids) to a list of strings that define top features for each sdg.
        """
        top_n = 1000
        pipe = self.__models['multiclass']

        # filtering out features not selected by the selector
        features = pipe['vectoriser'].get_feature_names_out()
        mask = pipe['selector'].get_support()
        features = features[mask]

        # extracting indices of top_n features from most to least important
        coefs = pipe['clf'].coef_
        mask = np.flip(coefs.argsort(axis = 1), axis = 1)[:, :top_n]

        # negative features should not be considered as (positive) predictors
        assert np.vstack([coefs[i, mask[i, -1]] for i in range(17)]).min(), 'All must be positive'

        # top top_n features per sdg
        features = features[mask]  # 17 by top_n
        sdg2features = {i + 1: features[i, :].tolist() for i in range(17)}
        return sdg2features

    def get_clean_features(self) -> Dict[int, List[str]]:
        """
        Extract features that are top predictions for each SDG from the multiclass classifier but do not
        repeat any feature in more than one sdg. If a feature appears in more than one sdg, the function returns it
        only for the sdg where it is ranked higher in the list of features.

        Returns
        -------
        sdg2features : Dict[int, List[str]]
            A mapping from integer numbers (sdg ids) to a list of strings that define top features for each sdg.
            Each feature is unique and appears only in the list for one sdg.
        """
        feature2sdg_rank = dict()
        sdg2features = defaultdict(list)
        for sdg, features in self.features.items():

            # for each feature consider its rank, as features are sorted by importance
            for rank, feature in enumerate(features):

                if feature not in feature2sdg_rank:
                    feature2sdg_rank[feature] = (sdg, rank)
                else:  # if the feature has already been seen, consider its rank
                    sdg_, rank_ = feature2sdg_rank[feature]  # previous highest rank
                    # if the current rank is lower than the best seen so far, replace it
                    if rank < rank_:
                        feature2sdg_rank[feature] = (sdg, rank)

        for feature, (sdg, _) in feature2sdg_rank.items():
            sdg2features[sdg].append(feature)
        return sdg2features
