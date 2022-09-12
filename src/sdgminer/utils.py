"""
Miscellaneous mapping and utilities for modifying and displaying sdg-related data.
"""
# standard library
import re
from typing import Iterable, Tuple, Any

# web services
import requests


# the list can be found at https://sdgs.un.org/goals
sdg_id2name = {
    1: 'Goal 1: No Poverty',
    2: 'Goal 2: Zero Hunger',
    3: 'Goal 3: Good Health and Well-being',
    4: 'Goal 4: Quality Education',
    5: 'Goal 5: Gender Equality',
    6: 'Goal 6: Clean Water and Sanitation',
    7: 'Goal 7: Affordable and Clean Energy',
    8: 'Goal 8: Decent Work and Economic Growth',
    9: 'Goal 9: Industry, Innovation and Infrastructure',
    10: 'Goal 10: Reduced Inequality',
    11: 'Goal 11: Sustainable Cities and Communities',
    12: 'Goal 12: Responsible Consumption and Production',
    13: 'Goal 13: Climate Action',
    14: 'Goal 14: Life Below Water',
    15: 'Goal 15: Life on Land',
    16: 'Goal 16: Peace and Justice Strong Institutions',
    17: 'Goal 17: Partnerships to achieve the Goal'
}

# from the sdg guidelines at https://www.un.org/sustainabledevelopment/news/communications-material/
sdg_id2color = {
    1: '#e5243b',
    2: '#DDA63A',
    3: '#4C9F38',
    4: '#C5192D',
    5: '#FF3A21',
    6: '#26BDE2',
    7: '#FCC30B',
    8: '#A21942',
    9: '#FD6925',
    10: '#DD1367',
    11: '#FD9D24',
    12: '#BF8B2E',
    13: '#3F7E44',
    14: '#0A97D9',
    15: '#56C02B',
    16: '#00689D',
    17: '#19486A'
}


def listify(l: Iterable[Any]) -> str:
    """
    A convenience routine to convert an iterable of objects into an unordered HTML list.

    Parameters
    ----------
    l : Iterable[Any]
        An iterable of elements, typically strings.

    Returns
    -------
    html_list : str
        A string corresponding to an unordered HTML list where each list item is an element from the original iterable.
    """
    html_list = '<ul>' + ''.join([f'<li>{x}</li>' for x in l]) + '</ul>'
    return html_list


def download_pdf(url: str, file_name: str) -> Tuple[bool, str]:
    """
    Download a .pdf document from a url to the local drive.

    Parameters
    ----------
    url : str
        A url to a .pdf file that must end with ".pdf".
    file_name : str
        An output name for the file

    Returns
    -------
    Tuple[bool, str]
        A True/False status indicating the outcome of the operation and a text message.
    """
    if not url.endswith('.pdf'):
        return False, 'The URL does not seem to refer to a .pdf file.'

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
            return True, 'Processing...'
    else:
        return False, response.text


def clean_whitespaces(text: str) -> str:
    """
    A convenience routine to standardise and replace repeating whitespace characters. This is useful for displaying
    text data in a user-friendly format.

    Parameters
    ----------
    text : str
        A text to be cleaned. If this is not a string, an empty string is returned.
    Returns
    -------
    text_ : str
        A cleaned copy of the original string.
    """
    if not isinstance(text, str):
        return ''

    text_ = re.sub(r'\s', ' ', text)  # handling space characters, incl. tab and return
    text_ = re.sub(r'\s{2,}', ' ', text_)  # removing repetitions
    return text_
