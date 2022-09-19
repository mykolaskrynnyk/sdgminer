"""
This module defines Manager classes which contain high-level abstractions for interacting with the package.
"""

# standard library
from collections import deque

# data wrangling
import pandas as pd

# local packages
from .extraction import TextExtractor
from .preprocessing import TextCleaner
from .transformations import calculate_sdg_salience
from .models import SDGModels

# utils
from tqdm import tqdm


class TextManager(object):
    def __init__(self, model_dir: str, model_version: str):
        self.__filepath = None
        self.__text_extractor = TextExtractor()
        self.__text_cleaner = TextCleaner()
        self.__models = SDGModels(path=model_dir, version=model_version)
        self.__sentences = None
        self.__paragraph_len = 1
        self.__df_texts = None

    def __repr__(self):
        return 'TextManager()'

    def __str__(self):
        return 'TextManager()'

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
                index = map(str, range(len(self.sentences))),
                columns = ['text', 'text_']
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

    def get_salience(self, model_type: str):
        assert model_type in {'multiclass', 'multilabel'}
        salience = calculate_sdg_salience(y_pred=self.__df_texts[f'sdgs_{model_type}'].tolist())
        return salience

    @property
    def sentences(self):
        return self.__sentences

    @property
    def df_texts(self):
        return self.__df_texts
