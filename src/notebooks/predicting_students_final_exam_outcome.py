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
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.ensemble import VotingClassifier
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
# %%capture -ns predicting_students_final_exam_outcome feature_table


def get_feature_table(max_date=500, code_presentation="2013J"):
    """Returns the feature table computed from the OULAD dataset."""
    assessments = (
        filter_by_module_presentation(oulad.assessments, "DDD", code_presentation)
        # Filter out assessments that are after the max_date.
        .query(f"date <= {max_date} or assessment_type == 'Exam'").set_index(
            "id_assessment"
        )
    )
    vle = (
        filter_by_module_presentation(oulad.student_vle, "DDD", code_presentation)
        .loc[:, ["id_student", "date", "sum_click"]]
        # Categorize the date field by assessment date.
        .assign(
            date=lambda df: pd.cut(
                df.date,
                [-26] + assessments.date.values.tolist(),
                labels=assessments.date.values,
            )
        )
        # Sum scores by date.
        .groupby(["id_student", "date"])
        .agg(np.sum)
        .reset_index()
        # Reshape the vle table.
        .pivot(index="id_student", columns="date", values="sum_click")
        # Rename columns
        .rename(
            columns={
                assessment.date: f"assessment_{i+1}_sum_click"
                if assessment.assessment_type != "Exam"
                else "final_exam_sum_click"
                for i, (_, assessment) in enumerate(assessments.iterrows())
            }
        )
        .drop("final_exam_sum_click", axis=1)
    )
    return (
        filter_by_module_presentation(oulad.student_info, "DDD", code_presentation)
        .loc[
            :,
            [
                "age_band",
                "gender",
                "id_student",
                "highest_education",
                "num_of_prev_attempts",
                "final_result",
            ],
        ]
        # Transform gender, age_band and highest_education to numeric values.
        .replace(
            {
                "age_band": {"0-35": 0.0, "35-55": 0.5, "55<=": 1.0},
                "gender": {"M": 0.0, "F": 1.0},
                "highest_education": {
                    "No Formal quals": 0.0,
                    "Lower Than A Level": 0.25,
                    "A Level or Equivalent": 0.5,
                    "HE Qualification": 0.75,
                    "Post Graduate Qualification": 1.0,
                },
            }
        )
        .set_index("id_student")
        # Filter out students who have unregistered from the course before the start.
        .join(
            filter_by_module_presentation(
                oulad.student_registration, "DDD", code_presentation
            )
            .set_index("id_student")
            .query("not date_unregistration < 0")
            .loc[:, []],
            how="right",
        )
        .join(vle)
        .join(
            assessments.join(oulad.student_assessment.set_index("id_assessment"))
            .reset_index()
            .pivot(index="id_student", columns="id_assessment", values="score")
            .rename(
                columns={
                    id_assessment: f"assessment_{i+1}_score"
                    if assessment.assessment_type != "Exam"
                    else "final_exam_score"
                    for i, (id_assessment, assessment) in enumerate(
                        assessments.iterrows()
                    )
                }
            )
        )
    )


feature_table = pd.concat(
    [get_feature_table(), get_feature_table(code_presentation="2014B")]
)
display(feature_table)

# %% [markdown]
# ### Pre-Processing
#
# #### Handling NAs
#
# We notice many missing values from the `final_exam_score` column in the selected
# feature table.

# %%
print(
    f"The feature table has {len(feature_table)} rows and the final exam score "
    f"column has {feature_table.final_exam_score.isna().sum()} rows with NAs "
    f"({100*feature_table.final_exam_score.isna().sum() / len(feature_table):.0f}%)."
)


# %% [markdown]
# This is explained in the original OULAD paper of Kuzilek et al.
# [\[KHZ17\]](../notebooks/first_descriptive_analysis.html#id1):
# ```
# Results of the final exam are usually missing (since they are scored and used for the
# final marking immediately at the end of the module).
# ```
#
# Therefore we use the `final_results` column to fill the missing final exam
# values and then remove the `final_results` column.
#
# Other columns containing missing values we fill with the value `-1`.


# %%
def fill_nas(feature_table_df):
    """Fills NAs in the `final_exam_score` column with `final_result` values,
    drops the `final_result` column and fills remaining NAs with the value `-1`.
    """

    mask = feature_table_df.final_exam_score.isna()
    feature_table_df.loc[mask, "final_exam_score"] = (
        feature_table_df[mask].final_result.isin(["Pass", "Distinction"]) * 40
    )
    return feature_table_df.drop(columns="final_result").fillna(-1)


feature_table = fill_nas(feature_table)
display(feature_table)

# %% [markdown]
# #### Splitting train/test data and Normalization
#
# Now we randomly split the feature table rows into a train (80%) and test (20%) table
# and, as in the work of Tomasevic et al., we scale and normalize the selected
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
RANDOM_STATE = 0


def normalized_train_test_split(feature_table_df):
    """Returns the normalized tain/test split computed from the feature table."""
    x_train_, x_test_, y_train_, y_test_ = train_test_split(
        feature_table_df.drop(columns="final_exam_score"),
        feature_table_df["final_exam_score"],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    # Scale scores per assessment and final_exam_score.
    assessment_score_labels = feature_table_df.columns.values[
        feature_table_df.columns.str.match(r"assessment_[0-9]+_score")
    ]
    x_train_.loc[:, assessment_score_labels] /= 100
    x_test_.loc[:, assessment_score_labels] /= 100
    y_train_ = (y_train_ / 100 >= 0.4).astype(int)
    y_test_ = (y_test_ / 100 >= 0.4).astype(int)

    # Scale the sum of clicks per assessment and number of attempts.
    columns_slice = feature_table_df.columns.values[
        feature_table_df.columns.str.match(r"assessment_[0-9]+_sum_click")
    ].tolist() + ["num_of_prev_attempts"]

    # Note: we fit the scaler only on the train data to avoid leaking information
    # from the test data.
    scaler = MinMaxScaler().fit(x_train_.loc[:, columns_slice])
    x_train_.loc[:, columns_slice] = scaler.transform(x_train_.loc[:, columns_slice])
    x_test_.loc[:, columns_slice] = scaler.transform(x_test_.loc[:, columns_slice])
    return (x_train_, x_test_, y_train_, y_test_)


x_train, x_test, y_train, y_test = normalized_train_test_split(feature_table)
display(x_train)

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
# %%capture -ns predicting_students_final_exam_outcome gs_scores
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
            "random_state": [RANDOM_STATE],
        },
    ],
    # Artificial Neural Networks
    MLPClassifier: [
        {
            "max_iter": [1000],
            "validation_fraction": [0.2],
            "hidden_layer_sizes": [(10,)],  # [(10,), (20,), (52, 10)],
            "random_state": [RANDOM_STATE],
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
            "random_state": [RANDOM_STATE],
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
            "random_state": [RANDOM_STATE],
        }
    ],
}


def get_grid_search_scores():
    """Returns the grid search scores."""
    classifier_score = {"classifier": [], "score": []}
    for classifier, hyperparameters in classifier_hyperparameters.items():
        gs_classifier = GridSearchCV(
            classifier(), hyperparameters, scoring="f1", n_jobs=-1
        )
        gs_classifier.fit(x_train, y_train)
        classifier_score["classifier"].append(classifier.__name__)
        classifier_score["score"].append(gs_classifier.score(x_test, y_test))

    return classifier_score


gs_scores = pd.DataFrame(get_grid_search_scores()).round(4)
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
#
# Let's start by taking a look at the assessment table for the selected courses.

# %%
oulad.assessments[
    (oulad.assessments.code_module == "DDD")
    & (oulad.assessments.assessment_type == "TMA")
    & (
        (oulad.assessments.code_presentation == "2013J")
        | (oulad.assessments.code_presentation == "2014B")
    )
].sort_values("date")


# %% [markdown]
# We note that each course module has six intermediary assessments.
#
# Next, we use the final submisssion `date` field to filter out assessment related
# information after a given date and repeat the same data preprocessing and
# classification process as done previously.
#
# We also add Voting and MultiCons ensemble methods to check whether they might improve
# current results.


# %%
# %%capture -ns predicting_students_final_exam_outcome scores
def get_train_test_assessments_by_day(day):
    """Returns the train/test feature table filtered by date."""

    filtered_feature_table = pd.concat(
        [get_feature_table(day), get_feature_table(day, code_presentation="2014B")]
    )
    filtered_feature_table = fill_nas(filtered_feature_table)
    return normalized_train_test_split(filtered_feature_table)


def get_scores_by_assessment_date():
    """Returns a DataFrame with f1 prediction scores for each classifier."""
    # pylint: disable=too-many-locals
    result = {}
    # We select the date such that both courses include the same amount of assessments
    # after the filter.
    for day in [25, 53, 88, 123, 165, 207]:
        result[day] = []
        x_train_, x_test_, y_train_, y_test_ = get_train_test_assessments_by_day(day)
        train_predictions = []
        predictions = []
        estimators = []
        for classifier, hyperparameters in classifier_hyperparameters.items():
            gs_classifier = GridSearchCV(
                classifier(), hyperparameters, scoring="f1", n_jobs=-1
            )
            gs_classifier.fit(x_train_, y_train_)
            estimators.append((classifier.__name__, gs_classifier))
            predictions.append(gs_classifier.predict(x_test_))
            train_predictions.append(gs_classifier.predict(x_train_))
            result[day].append(round(f1_score(y_test_, predictions[-1]), 4))

        # Voting Classifier
        voting = VotingClassifier(estimators=estimators, voting="soft")
        voting.fit(x_train_, y_train_)
        result[day].append(round(f1_score(y_test_, voting.predict(x_test_)), 4))

        # Searching for the best merging_threshold.
        max_score = 0
        multicons = None
        for merging_threshold in np.arange(0, 1, 0.05):
            consensus = MultiCons(
                similarity_measure="JaccardIndex",
                optimize_label_names=True,
                consensus_function="consensus_function_12",
                merging_threshold=merging_threshold,
            ).fit(train_predictions)
            score = f1_score(y_train_, consensus.labels_.astype(bool))
            if score > max_score:
                max_score = score
                multicons = consensus

        result[day].append(
            round(f1_score(y_test_, multicons.fit(predictions).labels_.astype(bool)), 4)
        )

    return pd.DataFrame(
        result,
        index=[clf.__name__ for clf in classifier_hyperparameters]
        + ["Voting", "MultiCons"],
    )


scores = get_scores_by_assessment_date()
display(Markdown("F1 score at different points in time:"))
display(scores)
