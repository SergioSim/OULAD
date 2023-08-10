"""Utility functions for common OULAD dataset treatments."""

from itertools import product
from typing import Any

from pandas import DataFrame


def filter_by_module_presentation(
    data: DataFrame, code_module, code_presentation, drop=True
) -> DataFrame:
    """Filters the `data` DataFrame by `code_module` and `code_presentation`.

    Args:
        data (DataFrame): The OULAD DataFrame with `code_module` and
            `code_presentation` columns.
        code_module (str): The `code_module` column value to filter.
        code_presentation (str): The `code_presentation` column value to filter.
        drop (bool): Whether to drop the `code_module` and `code_presentation`
            columns after filtering. By default is set to `True`.

    Returns:
        result (DataFrame): The filtered OULAD DataFrame.
    """
    result = data.loc[
        (data.code_module == code_module)
        & (data.code_presentation == code_presentation)
    ]
    if drop:
        return result.drop(["code_module", "code_presentation"], axis=1)
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
