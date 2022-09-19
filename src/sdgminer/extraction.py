"""
This module defines TextExtractor class for extracting text data from PDFs.
"""
# standard library
import os
import warnings

# data wrangling
import pypdfium2 as pdfium

# utils
from tqdm import tqdm


class TextExtractor(object):
    """
    Extract text from a .pdf document.

    This is a wrapper around pypdfium2 to extract text information from PDFs.

    Attributes
    ----------
    pages: list
        A list of texts where each element corresponds to a text from a single page.
    errors: dict
        A mapping from page numbers to error messages, if any were encountered.
    """
    def __init__(self):
        self.__filepath = None
        self.__text = None

    def __repr__(self) -> str:
        return 'TextExtractor()'

    def __str__(self) -> str:
        return f'TextExtractor for {self.__filepath} with {len(self.pages):,} pages.'

    def extract_text(self, filepath: str) -> bool:
        """
        Extract text from a .pdf document.

        Parameters
        ----------
        filepath : str
            A path to a pdf or txt file to be extracted.
        Returns
        -------
        self: TextExtractor
        """
        root, extension = os.path.splitext(filepath)

        # reset any data that is already contained
        self.__filepath = filepath
        if extension == '.pdf':
            page_texts = list()
            pdf = pdfium.PdfDocument(filepath)
            for idx, page in tqdm(enumerate(pdf)):
                try:
                    page_text = page.get_textpage().get_text()
                    page_texts.append(page_text)
                except:
                    warnings.warn(f'Failed to extract text from page {idx}.')
            self.__text = ' '.join(page_texts)
        elif extension == '.txt':
            with open(self.__filepath, 'r') as file:
                self.__text = file.read()
        else:
            raise ValueError(f'{extension} files are not supported. Supported extensions are .pdf and .txt')

        return self

    @property
    def text(self) -> str:
        """
        Concatenated text from all pages.

        Returns
        -------
        text: str
            A text from a file.
        """
        return self.__text
