# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] user_expressions=[]
# # Predicting students final exam outcome
#
# This section aims to predict the `student final exam outcome`
# (Pass (score >= 40) / Fail (score < 40)).
# We try to replicate the machine learning analysis techinques from the work of
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
from IPython.display import display
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

from oulad import get_oulad

oulad = get_oulad()


# %% [markdown] user_expressions=[]
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
# <table class="colwidths-auto table">
#     <thead>
#         <tr>
#             <th class="text-center head"><p>DEMOGRAPHIC</p></th>
#             <th class="text-center head"><p>ENGAGEMENT</p></th>
#             <th class="text-center head"><p>PERFORMANCE</p></th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="text-left">
#                 <ul>
#                     <li>gender</li>
#                     <li>highest_education</li>
#                     <li>age_band</li>
#                 </ul>
#             </td>
#             <td class="text-left">
#                 <ul>
#                     <li>sum of clicks per assessment</li>
#                 </ul>
#             </td>
#             <td class="text-left">
#                 <ul>
#                     <li>scores per assessment</li>
#                     <li>number of attempts</li>
#                     <li>final_exam score</li>
#                 </ul>
#             </td>
#         </tr>
#     </tbody>
# </table>


# %%
def get_feature_table(max_date=500, code_presentation="2013J"):
    """Returns the feature table computed from the OULAD dataset."""

    # Select all assessments from the module.
    assessments = oulad.assessments[
        (oulad.assessments.code_module == "DDD")
        & (oulad.assessments.code_presentation == code_presentation)
    ]

    # Filter out assessments that are after the max_date
    assessments = assessments[
        (assessments.date <= max_date) | (assessments.assessment_type == "Exam")
    ]

    # Filter relevant rows and columns from the student_vle table.
    vle = oulad.student_vle.loc[
        (oulad.student_vle.code_module == "DDD")
        & (oulad.student_vle.code_presentation == code_presentation),
        ["id_student", "date", "sum_click"],
    ]

    # Categorize the date field by assessment date.
    previous_date = None
    for date in assessments.date:
        if previous_date:
            vle.loc[(vle.date > previous_date) & (vle.date < date), "date"] = date
        else:
            vle.loc[vle.date < date, "date"] = date
        previous_date = date

    # Sum scores by date.
    vle = vle.groupby(["id_student", "date"]).agg(np.sum).reset_index()

    # Reshape the vle table.
    vle = vle.pivot(index="id_student", columns="date", values="sum_click")

    # Rename columns
    vle = vle.rename(
        columns={
            assessment.date: f"assessment_{i+1}_sum_click"
            if assessment.assessment_type != "Exam"
            else "final_exam_sum_click"
            for i, (_, assessment) in enumerate(assessments.iterrows())
        }
    ).drop("final_exam_sum_click", axis=1)

    return (
        oulad.student_info.loc[
            (oulad.student_info.code_module == "DDD")
            & (oulad.student_info.code_presentation == code_presentation),
            [
                "id_student",
                "gender",
                "highest_education",
                "age_band",
                "num_of_prev_attempts",
                # The `final_result` column is only used to fill missing `final_exam`
                # values, it should be removed from the training set.
                "final_result",
            ],
        ]
        .set_index("id_student")
        .join(vle)
        .join(
            oulad.student_assessment[
                oulad.student_assessment.id_assessment.isin(assessments.id_assessment)
            ]
            .pivot(index="id_student", columns="id_assessment", values="score")
            .rename(
                columns={
                    assessment.id_assessment: f"assessment_{i+1}_score"
                    if assessment.assessment_type != "Exam"
                    else "final_exam_score"
                    for i, (_, assessment) in enumerate(assessments.iterrows())
                }
            )
        )
    )


feature_table = pd.concat(
    [get_feature_table(), get_feature_table(code_presentation="2014B")]
)
display(feature_table)

# %% [markdown] user_expressions=[]
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


# %% [markdown] user_expressions=[]
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

    final_exam_score_nas = feature_table.final_exam_score.isna()
    feature_table_df.loc[final_exam_score_nas, "final_exam_score"] = (
        feature_table[final_exam_score_nas].final_result.isin(["Pass", "Distinction"])
        * 40
    )
    return feature_table.drop(columns="final_result").fillna(-1)


feature_table = fill_nas(feature_table)
display(feature_table)

# %% [markdown] user_expressions=[]
# #### Splitting train / test data and Normalization
#
# Now we randomly split the feature table rows into a train (80%) and test (20%) table
# and, as in the work of Tomasevic et al., we scale and nomalize the selected features:
#
# <table class="colwidths-auto table">
#     <thead>
#         <tr>
#             <th class="text-center head"><p>Feature</p></th>
#             <th class="text-center head"><p>Normalization</p></th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td class="text-left">
#                 <ul>
#                     <li>gender</li>
#                 </ul>
#             </td>
#             <td class="text-left">
#                 <ul>
#                     <li>0 = male</li>
#                     <li>1 = female</li>
#                 </ul>
#             </td>
#         </tr>
#         <tr>
#             <td class="text-left">
#                 <ul>
#                     <li>age_band</li>
#                 </ul>
#             </td>
#             <td class="text-left">
#                 <ul>
#                     <li>0.0 = 0-35</li>
#                     <li>0.5 = 35-55</li>
#                     <li>1.0 = 55&lt;=</li>
#                 </ul>
#             </td>
#         </tr>
#         <tr>
#             <td class="text-left">
#                 <ul>
#                     <li>highest_education</li>
#                 </ul>
#             </td>
#             <td class="text-left">
#                 <ul>
#                     <li>0.00 = No Formal quals</li>
#                     <li>0.25 = Lower Than A Level</li>
#                     <li>0.50 = A Level or Equivalent</li>
#                     <li>0.75 = HE Qualification</li>
#                     <li>1.00 = Post Graduate Qualification</li>
#                 </ul>
#             </td>
#         </tr>
#         <tr>
#             <td class="text-left">
#                 <ul>
#                     <li>number of attempts</li>
#                     <li>sum of clicks per assessment</li>
#                 </ul>
#             </td>
#             <td class="text-left">
#                 <ul>
#                     <li>0-N scaled to [0-1]</li>
#                 </ul>
#             </td>
#         </tr>
#         <tr>
#             <td class="text-left">
#                 <ul>
#                     <li>scores per assessment</li>
#                     <li>final_exam_score</li>
#                 </ul>
#             </td>
#             <td class="text-left">
#                 <ul>
#                     <li>0-100 scaled to [0-1]</li>
#                 </ul>
#             </td>
#         </tr>
#     </tbody>
# </table>

# %%
RANDOM_STATE = 0


def normalized_train_test_split(feature_table_df, is_for_classification=True):
    """Returns the normalized tain/test split computed form the feature table.

    If `is_for_classification` is set to true (default) the final_exam_score will be
    converted into two classes 0 (score < 40 == Fail) and 1 (score >= 40 == Pass).
    """

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
    y_train_ /= 100
    y_test_ /= 100
    if is_for_classification:
        y_train_ = (y_train_ >= 0.4).astype(int)
        y_test_ = (y_test_ >= 0.4).astype(int)

    # Transform gender, age_band and highest_education to numeric values.
    gender_map = {"M": 0, "F": 1}
    age_band_map = {"0-35": 0, "35-55": 0.5, "55<=": 1}
    highest_education_map = {
        "No Formal quals": 0,
        "Lower Than A Level": 0.25,
        "A Level or Equivalent": 0.5,
        "HE Qualification": 0.75,
        "Post Graduate Qualification": 1,
    }

    x_train_.loc[:, "gender"] = x_train_.loc[:, "gender"].map(gender_map)
    x_test_.loc[:, "gender"] = x_test_.loc[:, "gender"].map(gender_map)

    x_train_.loc[:, "age_band"] = x_train_.loc[:, "age_band"].map(age_band_map)
    x_test_.loc[:, "age_band"] = x_test_.loc[:, "age_band"].map(age_band_map)

    x_train_.loc[:, "highest_education"] = x_train_.loc[:, "highest_education"].map(
        highest_education_map
    )
    x_test_.loc[:, "highest_education"] = x_test_.loc[:, "highest_education"].map(
        highest_education_map
    )

    # Scale sum of click per assessment and number of attempts.
    columns_slice = feature_table_df.columns.values[
        feature_table_df.columns.str.match(r"assessment_[0-9]+_sum_click")
    ].tolist() + ["num_of_prev_attempts"]

    # Note: we fit the scaler only on the train data to avoid "leaking" information
    # from the test data.
    scaler = MinMaxScaler().fit(x_train_.loc[:, columns_slice])
    x_train_.loc[:, columns_slice] = scaler.transform(x_train_.loc[:, columns_slice])
    x_test_.loc[:, columns_slice] = scaler.transform(x_test_.loc[:, columns_slice])
    return (x_train_, x_test_, y_train_, y_test_)


x_train, x_test, y_train, y_test = normalized_train_test_split(feature_table)
display(x_train)

# %% [markdown] user_expressions=[]
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

for classifier, hyperparameters in classifier_hyperparameters.items():
    gs_classifier = GridSearchCV(classifier(), hyperparameters, scoring="f1", n_jobs=-1)
    gs_classifier.fit(x_train, y_train)
    print(
        f"{classifier.__name__}: score={gs_classifier.score(x_test, y_test):.4f} "
        f"best_parameters={gs_classifier.best_params_}"
    )


# %% [markdown] user_expressions=[]
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

# %% [markdown] user_expressions=[]
# We note that each course module has six intermediary assessments.
#
# Next, we use the final submisssion `date` field to filter out assessment related
# information after a given date and repeat the same data preprocessing and
# classification process as done previously.
#
# We also add Voting and MultiCons ensemble methods to check whether they might improve
# current results.

# %%
result = {}
# We select the date such as both courses include the same amount of assessments
# after the filter.
for day in [25, 53, 88, 123, 165, 207]:
    result[day] = []
    feature_table = pd.concat(
        [
            get_feature_table(day),
            get_feature_table(day, code_presentation="2014B"),
        ]
    )
    feature_table = fill_nas(feature_table)
    x_train, x_test, y_train, y_test = normalized_train_test_split(feature_table)
    print(f"Computing classification results at day={day:.0f}...")

    train_predictions = []
    predictions = []
    estimators = []
    for classifier, hyperparameters in classifier_hyperparameters.items():
        gs_classifier = GridSearchCV(
            classifier(), hyperparameters, scoring="f1", n_jobs=-1
        )
        gs_classifier.fit(x_train, y_train)
        estimators.append((classifier.__name__, gs_classifier))
        predictions.append(gs_classifier.predict(x_test))
        train_predictions.append(gs_classifier.predict(x_train))
        result[day].append(round(f1_score(y_test, predictions[-1]), 4))

    # Voting Classifier
    voting = VotingClassifier(estimators=estimators, voting="soft")
    voting.fit(x_train, y_train)
    result[day].append(round(f1_score(y_test, voting.predict(x_test)), 4))

    # MultiCons
    multicons_options = {
        "similarity_measure": "JaccardIndex",
        "optimize_label_names": True,
        "consensus_function": "consensus_function_12",
    }
    # Searching for the best merging_threshold.
    max_score = 0  # pylint: disable=invalid-name
    merging_threshold = -1  # pylint: disable=invalid-name
    for mt in np.arange(0, 1, 0.05):
        recommended_consensus = (
            MultiCons(**multicons_options, merging_threshold=mt)
            .fit(train_predictions)
            .labels_.astype(bool)
        )
        score = f1_score(y_train, recommended_consensus)
        if score > max_score:
            max_score = score
            merging_threshold = mt
    print(f"MultiCons: selected merging_threshold={merging_threshold:0.2f}")

    recommended_consensus = (
        MultiCons(**multicons_options, merging_threshold=merging_threshold)
        .fit(predictions)
        .labels_.astype(bool)
    )
    result[day].append(round(f1_score(y_test, recommended_consensus), 4))


classifier_names = [
    classifier.__name__ for classifier in classifier_hyperparameters
] + ["Voting", "MultiCons"]

result_df = pd.DataFrame(result, index=classifier_names)
print("\nF1 score at different points in time:")
display(result_df)
