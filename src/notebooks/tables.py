# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tables of the OULAD dataset
#
# Let's take a first look at the OULAD tables.

# %%
from IPython.display import display

from oulad import get_oulad

# Load the OULAD dataset in memory
oulad = get_oulad()

# %% [markdown] tags=[]
# ## Assessments

# %%
display(oulad.assessments)

# %% [markdown]
# ## Courses

# %%
display(oulad.courses)

# %% [markdown]
# ## Student Assessment

# %%
display(oulad.student_assessment)

# %% [markdown]
# ## Student Info

# %%
display(oulad.student_info)

# %% [markdown]
# ## Student Registration

# %%
display(oulad.student_registration)

# %% [markdown]
# ## Student VLE

# %%
display(oulad.student_vle)

# %% [markdown]
# ## VLE

# %%
display(oulad.vle)
