"""
This module defines typed entities used to increase consistency wihtin the package.
"""

# standard library
from typing import Dict, Optional

# data wrangling
import pandas as pd
from pydantic import BaseModel

# local packages
from .utils import SDGConverter


class SalienceRecord(BaseModel):
    dictionary: Dict[int, float]
    mode_type: Optional[str]

    @property
    def df(self):
        # fill in values for missing sdgs, if any exist
        sdg_converter = SDGConverter()
        df = pd.DataFrame(
            data=[(sdg, self.dictionary.get(sdg, 0.)) for sdg in range(1, 18)],
            columns=['sdg_id', 'salience']
        )
        df.sort_values('sdg_id', ignore_index=True, inplace=True)
        df['sdg_name'] = sdg_converter.ids2names(df['sdg_id'].tolist())
        df['sdg_color'] = sdg_converter.ids2colors(df['sdg_id'].tolist())
        df['sdg_id'] = df['sdg_id'].astype(str)  # cast to string for plotting
        return df