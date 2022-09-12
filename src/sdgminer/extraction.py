"""
This module defines TextExtractor class for extracting text data from PDFs.
"""
# standard library
import warnings

# data wrangling
from pdfminer.high_level import extract_text

# utils
from tqdm import tqdm


class TextExtractor(object):
    """
    Extract text from a .pdf document.

    This is a wrapper around pdfplumber to extract text information from pdfs.

    Parameters
    ----------
    filepath : str
        A path to a pdf file to be extracted.

    Attributes
    ----------
    pages: list
        A list of texts where each element corresponds to a text from a single page.
    errors: dict
        A mapping from page numbers to error messages, if any were encountered.
    """
    def __init__(self, filepath: str):
        assert filepath.endswith('.pdf'), '`filepath` must point to a .pdf.'
        self.__filepath = filepath
        self.__text = None

    def __repr__(self) -> str:
        return 'TextExtractor()'

    def __str__(self) -> str:
        return f'TextExtractor for {self.__filepath} with {len(self.pages):,} pages.'

    def extract_text(self) -> bool:
        """
        Extract text from a .pdf document.

        Returns
        -------
        self: TextExtractor
        """
        # TODO: use low-level API for extracting page by page
        self.__text = extract_text(self.__filepath)
        return self

    @property
    def text(self) -> str:
        """
        Concatenated text from all pages.

        Returns
        -------
        text: str
            A concatenated string of texts extracted from all pages.
        """
        if self.__text is None:
            warnings.warn(f'Text has not been extracted yet. Run `extract_text` first.')
        text = self.__text
        return text
