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

    This is a container-class to interact with pre-trained ML models.

    Parameters
    ----------
    path : str
        A path to the directory where models are stored.
    version : str
        A string version of the models to use.
    """
    def __init__(self, path: str, version: str):
        self.__models = {
            'multiclass': joblib.load(os.path.join(path, f'clf-logreg-multiclass-v{version}.joblib')),
            'multilabel': joblib.load(os.path.join(path, f'clf-mlp-multilabel-v{version}.joblib')),
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
        threshold: float, optional, default=.5
            A minimum probability threshold to assign an sdg label.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe each column corresponding to each model's predictions. Indices are aligned with the corpus.
        """
        df_probs = self.predict_with_models(corpus=corpus, threshold=threshold)
        return df_probs

    def predict_with_model(
            self,
            model_type: str,
            corpus: List[str],
            ids: List[Any] = None,
            threshold: float = .5) -> pd.DataFrame:
        """
        Predict sdg labels for a given corpus using either a multiclass or multilabel model.

        Uses one of the two pre-trained models to assign sdg labels to texts in the corpus. For each text provides a
        dict of predicted sdgs and their respective probabilities. Excludes any predictions that have a probability
        lower that the threshold.

        Parameters
        ----------
        model_type: {'multiclass', 'multilabel'}
            One of the two available model types.
        corpus : List[str]
            A list of texts to predict sdgs for.
        ids : List[Any], optional
            A list of ids to be used as indices in the output dataframe. If not specified, a range index is used.
        threshold : float, optional, default=.5
            A minimum probability threshold to assign an sdg label.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe with one column mapping sdgs to predicted probabilities for each text. Indices are taken
            from ids argument.
        """
        assert model_type in {'multiclass', 'multilabel'}
        assert 0 < threshold < 1, 'theshold must be in the range (0, 1).'
        if ids is None:
            ids = range(len(corpus))
        y_probs = self.__models[model_type].predict_proba(corpus)
        df_probs = pd.DataFrame(y_probs, columns=list(range(1, 18)))
        df_probs.insert(0, 'text_id', ids)
        df_probs = df_probs.melt(id_vars='text_id', var_name='sdg', value_name='prob')\
            .query('prob > @threshold').sort_values('prob', ascending=False)

        # in case no rows are left, return an empty DataFrame
        if df_probs.empty:
            df_probs = df_probs.set_index('text_id').reindex([model_type], axis=1)
        else:
            df_probs = df_probs.groupby('text_id')\
                .apply(lambda row: dict(zip(row['sdg'], row['prob'].round(2))))\
                .to_frame(name=model_type)
        return df_probs

    def predict_with_models(self, corpus: List[str], ids: List[Any] = None, threshold: float = .5) -> pd.DataFrame:
        """
        Predict sdg labels for a given corpus using both models.

        Uses each model to predict sdgs and store them in a column. If ids are passed, these are used as indices in
        the output dataframe. If not, a range index is used.

        Parameters
        ----------
        corpus : List[str]
            A list of texts to predict sdgs for.
        ids : List[Any], optional
            A list of ids to be used as indices in the output dataframe. If not specified, a range index is used.
        threshold : float, optional, default=.5
            A minimum probability threshold to assign an sdg label.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe each column corresponding to each model's predictions. Indices are aligned with the corpus.
        """
        if ids is not None:
            assert len(corpus) == len(ids), 'corpus and ids must be of the same length.'
        df_multiclass = self.predict_with_model(model_type='multiclass', corpus=corpus, ids=ids, threshold=threshold)
        df_multilabel = self.predict_with_model(model_type='multilabel', corpus=corpus, ids=ids, threshold=threshold)
        df_probs = df_multilabel.join(df_multiclass, how='outer')
        return df_probs

    def predict(self, corpus: List[str], ids: List[Any] = None, threshold: float = .5) -> pd.DataFrame:
        """
        An alias for self.predict_with_models

        Parameters
        ----------
        corpus : List[str]
            A list of texts to predict sdgs for.
        ids : List[Any], optional
            A list of ids to be used as indices in the output dataframe. If not specified, a range index is used.
        threshold : float, optional, default=.5
            A minimum probability threshold to assign an sdg label.

        Returns
        -------
        df_probs : pd.DataFrame
            A dataframe each column corresponding to each model's predictions. Indices are aligned with the corpus.
        """
        df_probs = self.predict_with_models(corpus=corpus, ids=ids, threshold=threshold)
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
        Extract features that are top predictors for each SDG from the multiclass classifier but do not
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
