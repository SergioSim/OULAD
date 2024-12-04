"""Utility functions for common OULAD dataset treatments."""

from itertools import product
from typing import Any

from pandas import DataFrame


def filter_by_module_presentation(
    data: DataFrame, code_module: str, code_presentation: str | tuple, drop: bool = True
) -> DataFrame:
    """Filters the `data` DataFrame by `code_module` and `code_presentation`.

    Args:
        data (DataFrame): The OULAD DataFrame with `code_module` and
            `code_presentation` columns.
        code_module (str): The `code_module` column value to filter.
        code_presentation (str or tuple): The `code_presentation` column value(s) to
            filter.
        drop (bool): Whether to drop the `code_module` and `code_presentation`
            columns after filtering.
            If filtering involves multiple `code_presentation` values the
            `code_presentation` columns is kept.
            By default is set to `True`.

    Returns:
        result (DataFrame): The filtered OULAD DataFrame.
    """
    match_code_module = data.code_module == code_module

    if isinstance(code_presentation, str):
        match_code_presentation = data.code_presentation == code_presentation
        drop_columns = ["code_module", "code_presentation"]
    else:
        match_code_presentation = data.code_presentation.isin(code_presentation)
        drop_columns = ["code_module"]

    result = data.loc[match_code_module & match_code_presentation]

    if drop:
        return result.drop(drop_columns, axis=1)

    return result


def grid_to_list(grid: dict[Any, dict[str, list]]) -> list[dict]:
    """Expands a parameter grid dictionary to a list of tuples.

    Args:
        grid (dict): The parameter grid to expand. Ex.:
            ```
            {
               "foo": {
                    "toto": [1, 2, 3],
               },
               "bar": {
                    "tata": [1, 4],
                    "titi": [0],
               }
            }
            ```

    Returns:
        result (list): A list of tuples. Ex.:
           ```
           [
                ("foo", {"toto": 1}),
                ("foo", {"toto": 2}),
                ("foo", {"toto": 3}),
                ("bar", {"tata": 1, "titi": 0}),
                ("bar", {"tata": 4, "titi": 0}),
           ]
           ```
    """
    result = []
    for key, parameters in grid.items():
        keys, values = zip(*parameters.items())
        for value in product(*values):
            result.append((key, dict(zip(keys, value))))
    return result
