"""Utility functions for common OULAD dataset treatments."""

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
