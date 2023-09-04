"""Loading the OULAD dataset into memory."""

import os
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd


@dataclass
class OULAD:
    """Represents the OULAD dataset tables."""

    # pylint: disable=too-many-instance-attributes

    assessments: pd.DataFrame
    courses: pd.DataFrame
    domains: pd.DataFrame
    student_assessment: pd.DataFrame
    student_info: pd.DataFrame
    student_registration: pd.DataFrame
    student_vle: pd.DataFrame
    vle: pd.DataFrame


@lru_cache(maxsize=1)
def get_oulad(path: str = None) -> OULAD:
    """Returns the OULAD dataset tables in a dataclass."""

    path = path if path else os.environ.get("OULAD_DEFAULT_PATH", "/app/OULAD")
    # Remove unlinked assessments (they don't have any corresponding student_assessment)
    assessments = pd.read_csv(f"{path}/assessments.csv")
    assessments.drop(
        assessments[assessments.id_assessment.isin([40088, 40087])].index, inplace=True
    )

    # Remove unlinked vle resources (they don't have any corresponding student_vle)
    # Also dropping `week_from` and `week_to` columns as most of the values are missing.
    vle = pd.read_csv(f"{path}/vle.csv")
    student_vle = pd.read_csv(f"{path}/studentVle.csv")
    resources = pd.Index(student_vle.id_site.unique())
    vle = vle.loc[
        ~vle.id_site.isin(resources.symmetric_difference(vle.id_site.values)),
        ["id_site", "code_module", "code_presentation", "activity_type"],
    ]

    return OULAD(
        assessments=assessments,
        courses=pd.read_csv(f"{path}/courses.csv"),
        domains=pd.DataFrame(
            {
                "code_module": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
                "domain": [
                    "Social Sciences",
                    "Social Sciences",
                    "STEM",
                    "STEM",
                    "STEM",
                    "STEM",
                    "Social Sciences",
                ],
            },
        ),
        student_assessment=pd.read_csv(f"{path}/studentAssessment.csv"),
        student_info=pd.read_csv(f"{path}/studentInfo.csv"),
        student_registration=pd.read_csv(f"{path}/studentRegistration.csv"),
        student_vle=student_vle,
        vle=vle,
    )
