"""
This module defines a TextCleaner class for preprocessing text input at different stages of the analysis.
"""

# standard Library
import re
import warnings
from typing import List, Tuple, Set, Union

# data wrangling
import numpy as np

# nlp
import spacy
from spacy.tokens import Span, Doc

# utils
from tqdm import tqdm


class TextCleaner(object):
    """
    Clean text data using s spaCy model.

    This is a wrapper around a spaCy model augmented with convenience methods for text cleaning.

    Parameters
    ----------
    allowed_pos : Set[str]
        A set of spaCy part-of-speech string tags to include in the cleaned document. Any token that has a pos tag
        which is not in this set is excluded.
    max_length : int
        Maximum sequence length is passed to a spaCy model. Sequences that are longer are truncated to max_length.
    Attributes
    ----------
    nlp : spacy.lang.en.English
        A English spacy model.
    """
    def __init__(self, allowed_pos: Set[str] = None, max_length: int = 1_000_000):
        self.nlp = spacy.load('en_core_web_sm', disable = ['ner'])
        self.nlp.max_length = max_length
        self.allowed_pos = allowed_pos or {'NOUN', 'VERB', 'ADJ'}

    def __repr__(self) -> str:
        return 'TextCleaner()'

    def __str__(self) -> str:
        version = 'SpaCy version: ' + spacy.__version__
        model = 'Model: ' + self.nlp.meta['name']
        discomponents = 'Disabled components: ' + str(self.nlp.disabled)
        pos = 'Allowed parts of speech: ' + str(self.allowed_pos)
        return f'{version}\n{model}\n{discomponents}\n{pos}'

    def __call__(self, texts: List[str]) -> List[str]:
        """
        Clean texts by only keeping tokens with POS tags that are in the allowed_pos set.

        This is a wrapper around TextCleaner.clean_texts().

        Parameters
        ----------
        texts: List[str]
            A list of texts to be cleaned.

        Returns
        -------
        texts_ : List[str]
            A list of texts where only tokens with POS tags are in the allowed list.
        """
        texts_ = self.clean_texts(texts)
        return texts_

    def clean_doc(self, doc: Union[Span, Doc]) -> str:
        """
        Clean a doc by removing tokens with POS that are not in the allowed list and lemmatising the tokens that are
        left.

        Parameters
        ----------
        doc : Union[Span, Doc]
            A spaCy Doc or Span obeject to be cleaned.

        Returns
        -------
        text_ : str
            A cleaned concatenated text of lemmatised tokens.
        """
        tokens = [token.lemma_ for token in doc if token.pos_ in self.allowed_pos and not token.is_stop]
        text_ = ' '.join(tokens)
        return text_

    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean a list of texts by first removing whitespaces and the applying TextCleaner.clean_doc().
        left.

        Parameters
        ----------
        texts : List[str]
            A list of input texts to be cleaned.

        Returns
        -------
        texts_ : List[str]
            A list of cleaned texts. See TextCleaner.clean_doc() for details.
        """
        texts = [self.clean_whitespaces(text) for text in texts]
        texts_ = list()
        for doc in tqdm(self.nlp.pipe(texts, batch_size = 128)):
            text_ = self.clean_doc(doc)
            texts_.append(text_)
        return texts_

    def sentencise(self, text: str) -> Tuple[str, str]:
        """
        A generator to split a text into sentences and clean each sentence afterwards.

        Parameters
        ----------
        text : str
            A list of input texts to be cleaned.

        Returns
        -------
        A generator of tuple where the first element is an original sentence text and the second element is its
        cleaned copy.
        """
        text_ = self.clean_whitespaces(text)
        if len(text_) > self.nlp.max_length:
            text_ = text_[:self.nlp.max_length]
            warnings.warn(f'The cleaned text is too long and has been truncated to {self.nlp.max_length:,} chars.')
        doc = self.nlp(text_)
        for sent in doc.sents:
            sent_ = self.clean_doc(sent)
            yield sent.text, sent_

    def vectorise(self, texts: List[str]) -> np.ndarray:
        """
        Get word-vectors for each text in the texts list.

        Parameters
        ----------
        texts : List[str]
            A list of texts to get embeddings for.

        Returns
        -------
        X : np.ndarray
            An array of size (n_texts, embed_dim).
        """
        X = np.vstack([doc.vector for doc in tqdm(self.nlp.pipe(texts, batch_size = 128))])
        return X

    @staticmethod
    def clean_whitespaces(text: str) -> str:
        """
        A convenience routine to standardise and replace repeating whitespace characters.

        Parameters
        ----------
        text : str
            A text to be cleaned. If this is not a string, an empty string is returned.

        Returns
        -------
        text_ : str
            A cleaned copy of the original string.
        """
        text_ = re.sub(r'\s', ' ', text)  # handling space characters, incl. tab and return
        text_ = re.sub(r'\s{2,}', ' ', text_)  # removing repetitions
        return text_
