"""
This module defines TextExtractor class for extracting text data from PDFs.
"""
# standard library
import warnings

# data wrangling
import pypdfium2 as pdfium

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
        self.__pages = dict()
        self.__errors = list()

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
        pdf = pdfium.PdfDocument(self.__filepath)
        for idx, page in tqdm(enumerate(pdf)):
            try:
                self.__pages[idx] = page.get_textpage().get_text()
            except:
                self.__errors.append(idx)
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
        if len(self.__pages) == 0:
            warnings.warn(f'Text has not been extracted yet. Run `extract_text` first.')
        if len(self.__errors) > 0:
            warnings.warn(f'Extraction from {len(self.__errors)} pages has failed, e.g., {self.__errors[:5]}')
        text = ' '.join(self.__pages.values())
        return text
