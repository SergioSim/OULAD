"""Loading the OULAD dataset into memory."""

from dataclasses import dataclass

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


def get_oulad(path: str = "/app/OULAD") -> OULAD:
    """Returns the OULAD dataset tables in a dataclass."""

    return OULAD(
        assessments=pd.read_csv(f"{path}/assessments.csv"),
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
        student_vle=pd.read_csv(f"{path}/studentVle.csv"),
        vle=pd.read_csv(f"{path}/vle.csv"),
    )
