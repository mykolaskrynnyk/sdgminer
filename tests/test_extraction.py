# standard library
import os
import re

# sdgminer
from sdgminer.extraction import TextExtractor


def test_extract():
    """
    Test text extraction from .pdf files.
    """
    with open(os.path.join('tests', 'lorem_ipsum.txt'), 'r') as file:
        text_source = file.read()
    text_source_sents = re.split(r'\n+|\.\s+', text_source)  # split titles and paragraphs into sentences

    # test typical one-column and two-column texts
    test_files = ('lorem_ipsum_1.pdf', 'lorem_ipsum_2.pdf')

    for file_name in test_files:
        text_extractor = TextExtractor(filepath = os.path.join('tests', file_name))
        text_extractor.extract_text()
        text_extracted = re.sub(pattern  = r'\s+', repl = ' ', string = text_extractor.text)
        match_count = sum(1 for sent in text_source_sents if sent in text_extracted)
        overlap = match_count / len(text_source_sents)
        assert overlap > .9, f'{file_name} failed with overlap = {overlap:.2f}'