# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A first descriptive analysis
#
# One of the best ways to get started with the OULAD analysis might be to explore the
# original paper that introduced the OULAD dataset.
# {cite}`kuzilek_hlosta_zdrahal_2017`
#
# In this section we try to reproduce and summarize their findings.
# We also take some notes at the end which might be used later.
#
# ```{bibliography}
# :filter: docname in docnames
# ```

# %%
import matplotlib.pyplot as plt
import pandas as pd

from oulad import get_oulad

oulad = get_oulad()

# %% [markdown]
# ## General statistics

# %%
module_count = oulad.courses.code_module.nunique()
print(
    "OULAD contains data about:\n"
    f"  - {oulad.courses.shape[0]} courses from {module_count} modules "
    "(4 STEM modules and 3 Social Sciences modules)\n"
    f"  - {oulad.student_info.shape[0]} students\n"
    f"  - {oulad.student_registration.shape[0]} student registrations\n"
    f"  - {oulad.student_vle.shape[0]} VLE interaction entries"
)

# %% [markdown]
# ### Student registration count by module with domain information

# %%
registration_count = (
    oulad.student_registration.groupby(
        ["code_module", "code_presentation"], as_index=False
    )
    .count()
    .groupby(["code_module"])
    .agg(
        presentations=pd.NamedAgg(column="code_presentation", aggfunc="count"),
        students=pd.NamedAgg(column="id_student", aggfunc="sum"),
    )
)
oulad.domains.join(registration_count, on="code_module")

# %% [markdown]
# ### Student registration count by module-presentation

# %%
registration_count = oulad.student_registration.groupby(
    ["code_module", "code_presentation"]
).size()
registration_count.reset_index()

# %%
max_id = registration_count.idxmax()
min_id = registration_count.idxmin()
print(
    f"The largest module-presentation {max_id} contains "
    f"{registration_count[max_id]} student registrations.\n"
    f"The smallest module-presentation {min_id} contains "
    f"{registration_count[min_id]} student registrations. \n"
    f"The average module-presentation registration count is "
    f"{registration_count.mean()}."
)

# %% [markdown]
# ### Student assessment count

# %%
exams = oulad.assessments[oulad.assessments.assessment_type == "Exam"]
print(
    f"The student_assessment table contains {oulad.student_assessment.shape[0]} rows."
    "\n"
    f"The assessment tabel contains {exams.shape[0]} Exams.\n"
    f"{pd.merge(oulad.student_assessment, exams, on='id_assessment').shape[0]} "
    "student_assessments are Exams."
)

# %% [markdown]
# ### Student info attributes distribution for CCC module

# %%
ccc_student_info = oulad.student_info[oulad.student_info.code_module == "CCC"].drop(
    "code_module", axis=1
)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18), constrained_layout=True)

ccc_student_info.groupby(["age_band", "final_result"]).size().unstack().plot.bar(
    stacked=True, ax=axes[0, 0], title="Student count by age_band"
)
ccc_student_info.groupby(["disability", "final_result"]).size().unstack().plot.bar(
    stacked=True, ax=axes[0, 1], title="Student count by disability"
)
ccc_student_info.groupby(
    ["highest_education", "final_result"]
).size().unstack().plot.bar(
    stacked=True, ax=axes[1, 0], title="Student count by highest_education"
)
ccc_student_info.groupby(["gender", "final_result"]).size().unstack().plot.bar(
    stacked=True, ax=axes[1, 1], title="Student count by gender"
)
ccc_student_info.groupby(["imd_band", "final_result"]).size().unstack().plot.bar(
    stacked=True, ax=axes[2, 0], title="Student count by imd_band"
)
ccc_student_info.groupby(["region", "final_result"]).size().unstack().plot.bar(
    stacked=True, ax=axes[2, 1], title="Student count by region"
)
plt.show()

# %% [markdown]
# ## Notes
#
# - The initial total number of students in the selected modules was 38239.
# - Students in a module presentation are organized into study groups of ~20 people.
# - Module resources are available from the VLE system a few weeks before the start.
# - If the final exam `date` is missing in the `assessments` table, it takes place
#   during the last week of the module presentation.
# - The structure of B and J presentations may differ.
# - In the `student_registration` table, the student has withdrawn if the
#   `date_unregistration` field is present.
# - If the student does not submit an assessment, no result is recorded.
# - The results of the final exam are usually missing.
# - An assessment score lower than 40 is interpreted as a failure.
