# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predicting student's final exam outcome
#
# This section aims to predict the `student final exam outcome`
# (Pass (score >= 40) / Fail (score < 40)).
# We try to replicate the machine learning analysis techniques from the work of
# Tomasevic et al. (2020) {cite}`tomasevic_2020`.
#
# **Keywords**: Predicting student outcome
#
# ```{bibliography}
# :filter: docname in docnames
# ```

# %%
from itertools import combinations
from functools import reduce

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from multicons import MultiCons

from oulad import filter_by_module_presentation, get_oulad

# %load_ext oulad.capture
pd.set_option("display.max_columns", 23)


# %%
# %%capture oulad
oulad = get_oulad()

# %% [markdown]
# ## Preparing train/test data
#
# ### Selecting features
#
# In the work of Tomasevic et al. the student data from the `DDD` module of the
# `2013J` and `2014B` presentations combined is used.
#
# Similarly, we try to select the same seven distinct attributes from the three distinct
# types below:
#
# | DEMOGRAPHIC         | ENGAGEMENT                     | PERFORMANCE             |
# |---------------------|--------------------------------|-------------------------|
# | - gender            | - sum of clicks per assessment | - scores per assessment |
# | - highest_education |                                | - number of attempts    |
# | - age_band          |                                | - final_exam score      |


# %%
CODE_MODULE = "DDD"
CODE_PRESENTATIONS = ("2013J", "2014B")

# %% [markdown]
# #### Demographic
#
# In this section we select the gender, highest education level and age band features from
# students in the `DDD` course.

# %%
student_info = (
    filter_by_module_presentation(oulad.student_info, CODE_MODULE, CODE_PRESENTATIONS)
    .set_index(["id_student", "code_presentation"])
)
demographic = student_info.loc[:, ["gender", "highest_education", "age_band"]]
display(Markdown("### Demographic"))
display(demographic)

# %% [markdown]
# We note that in total 3166 students have enrolled one of the two `DDD` presentations.

# %% [markdown]
# #### Performance
#
# In this section we select the score for each assessment, the final exam score and the
# number of attempts the student made.

# %%
attempts = student_info.loc[:, ["num_of_prev_attempts"]]
assessments = (
    filter_by_module_presentation(oulad.assessments, CODE_MODULE, CODE_PRESENTATIONS)
    # The DDD2013J assessments and exams start with ID 25348 and end with 25354.
    # The DDD2014B assessments and exams start with ID 25355 and end with 25361.
    # Thus, we remove 25347 from the ID to make them between 1 and 14.
    .assign(assessment=lambda df: df.id_assessment - 25347)
    # Now, we align the assessments and exams of both presentations.
    # We want the first assessment in both presentations to have the ID 1,
    # the second assessment to have the ID 2, etc.
    .assign(assessment=lambda df: df.assessment - (df.assessment > 7) * 7)
    .set_index("id_assessment")
)
student_assessment = (
    oulad.student_assessment.set_index("id_assessment")
    .join(assessments, how="right")
    .reset_index()
    .pivot_table(
        index=["id_student", "code_presentation"],
        columns="assessment",
        values="score",
        aggfunc="sum"
    )
    .rename(columns=lambda x: f"assessment_{x}_score" if x < 7 else "final_exam_score")
    # Remove students that have no final exam score.
    # .pipe(lambda df: df[df.final_exam_score.notna()])
)
performance = attempts.join(student_assessment)
display(Markdown("### Performance"))
display(performance)

# %% [markdown]
# We note that some of the students have no final exam score (NaN).

# %% [markdown]
# #### Engagement
#
# In this section we compute the sum of clicks per assessment.

# %%
engagement = (
    filter_by_module_presentation(oulad.student_vle, CODE_MODULE, CODE_PRESENTATIONS)
    .drop(columns="id_site")
    .merge(
        assessments.pivot(index="code_presentation", columns="date", values="date"),
        on="code_presentation"
    )
    .assign(
        date=lambda df: (
            (df.iloc[:, 4:].sub(df.date, axis="index"))
            .pipe(lambda df: df.where(df >= 0))
            .idxmin(axis=1)
            .astype(int)
        )
    )
    .replace({"date": assessments.groupby(["date"])["assessment"].first().to_dict()})
    .pivot_table(
        index=["id_student", "code_presentation"],
        columns="date",
        values="sum_click",
        aggfunc="sum",
        fill_value=0,
    )
    .rename(columns=lambda col: f"clicks_{col}")
    .join(student_info.loc[:, []], how="right")
)
display(Markdown("### Engagement"))
display(engagement)

# %% [markdown]
# We note that some students have no interactions with the VLE.

# %% [markdown]
# ### Master table
#
# We join the demographic, performance and engagement data into one table.

# %%
master_table = demographic.join(performance, how="inner").join(engagement, how="inner")
display(Markdown("### Master table"))
display(master_table)

# %% [markdown]
# ### Pre-Processing
#
# #### Handling NaNs
#
# We check the master table columns for missing values.

# %%
student_count = master_table.shape[0]
display(Markdown("### Master table NaNs count and percentage"))
display(
    master_table
    .isna()
    .sum()
    .rename("nan_count")
    .to_frame()
    .assign(nan_percentage=lambda df: (100 * df.nan_count / student_count).round(2))
)

# %% [markdown]
# More than the half of `final_exam_scores` are missing.
#
# The high number of missing exam scores is explained in the original OULAD paper
# of Kuzilek et al. \[[KHZ17](first_descriptive_analysis)\]:
#
# > Results of the final exam are usually missing (since they are scored and used for the
# > final marking immediately at the end of the module).
#
# Therefore we check the final result repartition of students that have no final exam
# score.

# %%
display(
    master_table
    .join(student_info[["final_result"]])
    .pipe(lambda df: df[df.final_exam_score.isna()])
    .final_result
    .value_counts()
    .rename("Students without final exam score")
    .to_frame()
)

# %% [markdown]
# We note that most student either have failed or have withdrawn from the course.
# Surprisingly, one student has passed the course.
# We take a closer look on the student record and compare it with an average student
# record below.

# %%
display(Markdown("The student record that passed the couse without a final exam score"))
display(
    master_table
    .join(student_info[student_info.columns.difference(master_table.columns)])
    .pipe(lambda df: df[df.final_result == "Pass"])
    .pipe(lambda df: df[df.final_exam_score.isna()])
)
display(Markdown("Average student performance and engagement by final result"))
display(
    master_table
    .join(student_info[student_info.columns.difference(master_table.columns)])
    .replace(
        {
            "final_result": {
                "Pass": "2",
                "Distinction": "3",
                "Fail": "1",
                "Withdrawn": "0",
            }
        }
    )
    .astype({"final_result": int})
    .select_dtypes(include="number")
    .groupby("final_result")
    .agg("mean")
    .reset_index()
    .astype({"final_result": str})
    .replace(
        {
            "final_result": {
                "2": "Pass",
                "3": "Distinction",
                "1": "Fail",
                "0": "Withdrawn",
            }
        }
    )
)

# %% [markdown]
# We observe that the performance and engagement mertics for the student that passed the
# course without having a final exam score are close to the average values of other
# passing students.
#
# Thus we decided to exlude this student record from the dataset as it represents an
# outlier: a rare case of passing without a final exam score, which deviates from the
# typical pattern of failure or withdrawal.
#
# Next, some students have withrawn from the course before the course start.

# %%
students_withdrawn_before_start = (
    filter_by_module_presentation(
        oulad.student_registration, CODE_MODULE, CODE_PRESENTATIONS
    )
    .set_index(["id_student", "code_presentation"])
    .join(master_table)
    .pipe(lambda df: df[df.date_unregistration <= 0])
    .index
)
count = len(students_withdrawn_before_start)
display(Markdown(f"{count} students withrew before course start"))

# %% [markdown]
# These students can be removed from classification dataset as it is known at course start
# that these students would not attempt the final exam.
#
# We also remove those student that have made no interactions with the vle.
#
# For the remaining missing values we choose to fill them with zeros.

# %%
master_table_filtered = (
    master_table
    # Drop passing student without final exam score.
    .drop((592315, "2013J")) 
    .drop(students_withdrawn_before_start)
    # Drop students without any vle interactions.
    .pipe(lambda df: df[~df[engagement.columns].isna().all(axis=1)])
    .fillna(0)
)
display(Markdown("### Master table after handling missing values"))
display(master_table_filtered)

# %% [markdown]
# #### Normalization
#
# As in the work of Tomasevic et al., we scale and normalize the selected
# features:
#
# ```{list-table}
# :header-rows: 1
#
# *   - Feature
#     - Normalization
#
# *   - Gender
#     - 0 = male
#
#       1 = female
#
# *   - Age band
#     - 0.0 = 0-35
#
#       0.5 = 35-55
#
#       1.0 = 55<=
#
# *   - Highest education
#     - 0.00 = No Formal quals
#
#       0.25 = Lower Than A Level
#
#       0.50 = A Level or Equivalent
#
#       0.75 = HE Qualification
#
#       1.00 = Post Graduate Qualification
#
# *   - Number of attempts
#
#       Sum of clicks per assessment
#     - 0-N scaled to [0-1]
#
# *   - Scores per assessment
#
#       Final exam score
#     - 0-100 scaled to [0-1]
# ```

# %%
master_table_normalized = (
    master_table_filtered
    # Normalize demographic data.
    .replace(
        {
            "age_band": {"0-35": "0.0", "35-55": "0.5", "55<=": "1.0"},
            "gender": {"M": "0.0", "F": "1.0"},
            "highest_education": {
                "No Formal quals": "0.0",
                "Lower Than A Level": "0.25",
                "A Level or Equivalent": "0.5",
                "HE Qualification": "0.75",
                "Post Graduate Qualification": "1.0",
            },
        }
    )
    .astype(float)
    .pipe(
        lambda df: pd.DataFrame(
            MinMaxScaler().fit_transform(df).round(2),
            columns=df.columns,
            index=df.index
        )
    )
)
display(Markdown("### Master table normalized"))
display(master_table_normalized)

# %% [markdown]
# #### Discretisation
#
# We want to classify students into two categories: `Fail` (0) and `Pass` (1),
# based on their final exam scores, where a score below 40 is classified as "Fail"
# and a score of 40 or higher as "Pass."

# %%
# The `final_exam_score` was normalized using the MinMaxScaler.
# The MinMaxScaler uses the follwing formula:
# > X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# > X_scaled = X_std * (max - min) + min
# We kept the default max = 1 and min = 0 values, thus X_scaled = X_std.
get_separator = lambda x: (40 - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
separator = get_separator(master_table_filtered.final_exam_score) 
final_exam_score_classes = (
    (master_table_normalized.final_exam_score >= separator).astype(int)
)
final_exam_score_classes.value_counts().to_frame()

# %% [markdown]
# #### Splitting train/test data
#
# Now we randomly split the normalized and discretisized master table rows into a train
# (80%) and test (20%) table.
#
# We also partition the train table into a train (60%) and validation (20%) set for
# algorithms that require a validation set e.g. Neural Networks.
#
# We notice a class imbalance - the `Pass` class appears much more frequently than the
# `Fail` class.
# Thus to avoid contructing a train or test subset without the `Fail` class, we choose
# to make a stratified split (one that keeps the original proportions for each class).

# %%
x_train_80, x_test_20, y_train_80_class, y_test_20_class = train_test_split(
    master_table_normalized.drop(columns="final_exam_score"),
    final_exam_score_classes,
    test_size=0.2,
    stratify=final_exam_score_classes
)
x_train_60, x_validate_20, y_train_60_class, y_validate_20_class = train_test_split(
    x_train_80, y_train_80_class, test_size=0.25, stratify=y_train_80_class
)
get_regression_values = lambda y: master_table_normalized.final_exam_score[y.index]
y_train_80 = get_regression_values(y_train_80_class)
y_test_20 = get_regression_values(y_test_20_class)
y_train_60 = get_regression_values(y_train_60_class)
y_validate_20 = get_regression_values(y_validate_20_class)


# %% [markdown]
# ## Classification
#
# As in the work of Tomasevic et al., we will compare the classification performances
# for the student final exam pass prediction (score >= 40).
#
# We use the same models and try to perform a grid search over the same Hyper-parameter
# ranges if these were specified in the paper:
#
# - K-Nearest Neighbours (with & without `weights`, varying `K` between 1 and 50)
# - Support Vector Machines (with `linear` and `RBF` kernels, varying `C` in
# `[0.1, 1.0, 10]`, varying gamma in `[0.0001, 0.01, 0.1]`)
# - Artificial Neural Networks (with one and two hidden layers)
# - Decision Trees (with varying `max depth`, `split` strategy and `quality measure`)
# - Naïve Bayes (with varying `var_smoothing`)
# - Logistic Regression (with `lbfgs` and `saga` solvers)
#
# And the performance metric used here is also the F1 score.
#
# As a reminder, the formula of the F1 score is:
# 2 * (precision * recall) / (precision + recall)
#
# However, in contrast to the paper, we use 5-fold cross validation during the grid
# search phase.

# %%
# # %%capture -ns predicting_students_final_exam_outcome gs_scores
# Hyperparameter search space

classifier_hyperparameters = {
    # K-Nearest Neighbours
    KNeighborsClassifier: [
        # {"n_neighbors": range(1, 51), "weights":["uniform", "distance"]}
        # We reduce search space for speed
        {
            "n_neighbors": [24],
            "weights": ["distance"],
        }
    ],
    # Support Vector Machines
    SVC: [
        # {
        #     "kernel": ["linear"],
        #     "C": [0.1, 1.0, 10],
        #     "probability": [True],
        #     "random_state": [RANDOM_STATE],
        # },
        {
            "kernel": ["rbf"],
            "C": [10],  # [0.1, 1.0, 10],
            "gamma": ["scale"],  # ["scale", "auto", 0.0001, 0.01, 0.1],
            "probability": [True],
        },
    ],
    # Artificial Neural Networks
    MLPClassifier: [
        {
            "max_iter": [1000],
            "validation_fraction": [0.2],
            "hidden_layer_sizes": [(10,)],  # [(10,), (20,), (52, 10)],
            # [(i,) for i in range(2, 100, 10)] + [
            #     (i, j) for i in range(2, 100, 10) for j in range(2, 100, 10)
            # ],
            # As we do not notice any improvement by varying `activation` and `alpha`,
            # we choose to keep the default values for these parameters.
            # "activation": ["logistic", "tanh", "relu"],
            # "alpha": 10.0 ** (- np.arange(-1,6))
        },
    ],
    # Decision Tree
    DecisionTreeClassifier: [
        {
            "criterion": ["entropy"],  # ["gini", "entropy"],
            "splitter": ["best"],  # ["best", "random"],
            "max_depth": [6],  # [None, *list(range(1, 11))],
            "min_samples_split": [2],  # range(2, 11, 2),
            "min_samples_leaf": [10],  # range(2, 11, 2),
        },
    ],
    # Naive Bayes
    GaussianNB: [
        {
            "var_smoothing": [1e-9],  # [1e-9, 1e-8, 1e-7, 1e-6]
        }
    ],
    # Logistic Regression
    LogisticRegression: [
        {
            "solver": ["lbfgs"],  # ["lbfgs", "saga"],
        }
    ],
}

def get_grid_search_scores(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    hyperparameters: dict[BaseEstimator, list[dict[str, list]]],
    title: str
) -> pd.DataFrame:
    """Returns the grid search scores."""
    classifier_score = {"classifier": [], title: []}
    estimators = []
    train_predictions = []
    predictions = []
    for classifier, hyperparameter in hyperparameters.items():
        gs_classifier = GridSearchCV(
            classifier(), hyperparameter, scoring="f1", n_jobs=-1
        )
        gs_classifier.fit(x_train, y_train)
        estimators.append((classifier.__name__, gs_classifier))
        train_predictions.append(gs_classifier.predict(x_train))
        predictions.append(gs_classifier.predict(x_test))
        classifier_score["classifier"].append(classifier.__name__)
        classifier_score[title].append(round(f1_score(y_test, predictions[-1]), 4))
    

    # Voting Classifier
    voting = VotingClassifier(estimators=estimators, voting="soft")
    voting.fit(x_train, y_train)
    classifier_score["classifier"].append("Voting")
    classifier_score[title].append(round(f1_score(y_test, voting.predict(x_test)), 4))

    # Bagging Classifier
    bagging = BaggingClassifier()
    bagging.fit(x_train, y_train)
    classifier_score["classifier"].append("Bagging")
    classifier_score[title].append(round(f1_score(y_test, bagging.predict(x_test)), 4))
    
    # Stacking Classifier
    stacking = StackingClassifier(estimators=estimators)
    stacking.fit(x_train, y_train)
    classifier_score["classifier"].append("Stacking")
    classifier_score[title].append(round(f1_score(y_test, stacking.predict(x_test)), 4))
    
    # SupMultiCons Classifier
    # Searching for the best merging_threshold.
    max_score = 0
    multicons = None
    consensus_functions = [
        "consensus_function_12",
        "consensus_function_13",
        "consensus_function_14",
        "consensus_function_15",
    ]
    for consensus_function in consensus_functions:
        for merging_threshold in np.arange(0, 1, 0.05):
            consensus = MultiCons(
                similarity_measure="JaccardIndex",
                optimize_label_names=True,
                consensus_function=consensus_function,
                merging_threshold=merging_threshold,
            ).fit(train_predictions)
            score = f1_score(y_train, consensus.labels_.astype(bool))
            if score > max_score:
                max_score = score
                multicons = consensus

    classifier_score["classifier"].append("SupMultiCons")
    classifier_score[title].append(
        round(f1_score(y_test, multicons.fit(predictions).labels_.astype(bool)), 4)
    )

    return pd.DataFrame(classifier_score).set_index("classifier").round(4)


columns = {
    "D": list(demographic.columns),
    "E": list(engagement.columns),
    "P": list(performance.columns.difference(["final_exam_score"]))
}
columns_subsets = {
    **columns,
    "D+E": columns["D"] + columns["E"],
    "D+P": columns["D"] + columns["P"],
    "E+P": columns["E"] + columns["P"],
    "D+E+P": columns["D"] + columns["E"] + columns["P"]
}

gs_scores = reduce(
    lambda a, b: a.join(b),
    [
        get_grid_search_scores(
            x_train_80[subset],
            y_train_80_class,
            x_test_20[subset],
            y_test_20_class,
            classifier_hyperparameters,
            column,
        )
        for column, subset in columns_subsets.items()
    ]
)
display(Markdown("### Comparison of F1 score for final exam classification"))
display(Markdown("(D - demographics, E - engagement, P - performance) data"))
display(gs_scores)


# %% [markdown]
# ### Classification at different points in time
#
# Predicting student final exam outcome seems to be more valuable at an early stage of
# the course as it might give instuctors more time to help the students at risk.
# However, predicting early is more challenging as less data is available for the
# classifiers.
#
# As in the work of Tomasevic et al., we will compare the classification performances at
# different moments of the course based on the number of assessments passed.

# %%
columns_subsets = {
    "After 1st assessment": columns["D"] + columns["E"][:1] + columns["P"][:1],
    "After 2nd assessment": columns["D"] + columns["E"][:2] + columns["P"][:2],
    "After 3rd assessment": columns["D"] + columns["E"][:3] + columns["P"][:3],
    "After 4th assessment": columns["D"] + columns["E"][:4] + columns["P"][:4],
    "After 5th assessment": columns["D"] + columns["E"][:5] + columns["P"][:5],
    "After 6th assessment": columns["D"] + columns["E"][:6] + columns["P"][:6],
}

gs_scores = reduce(
    lambda a, b: a.join(b),
    [
        get_grid_search_scores(
            x_train_80[subset],
            y_train_80_class,
            x_test_20[subset],
            y_test_20_class,
            classifier_hyperparameters,
            column,
        )
        for column, subset in columns_subsets.items()
    ]
)
display(
    Markdown(
        "### Comparison of F1 score for final exam classification"
        "at different points in time"
    )
)
display(gs_scores)
