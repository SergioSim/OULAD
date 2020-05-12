# Analysis of the OULAD dataset


## Motivation and Problem statement

Since 2011, online education in form of MOOCs (Massive Open Online Courses) is growing in popularity. The concept of educational material made available for free (or almost) attracted millions of people all over the world. MOOCs have the potential to improve the way we are learning online, reduce the cost of education, make education accessible and much more. But they suffer from one major drawback - high failure / withdrawal rates. Even though this seems to be an obvious side effect due to the high number of people starting a MOOC and their heterogeneity in terms of end goal, there is evidence [^1] that many of them actually intended to complete a course but weren't able to. A system that could predict if a MOOC student will fail/withdraw during the course and alert the pedagogical team would be a valuable tool for teachers, allowing them to take action in order to help students accomplish their goals and reduce the failure / withrawal rates.

[^1] : Chuang, I. & Ho, A. (2016). HarvardX and MITx: Four Years of Open Online Courses -- Fall 2012 - Summer 2016

## Scope

One major advantage MOOCs have in comparaison to traditional courses at bricks-and-mortar institutions is the quantity and granularity of data that they generate and record. This data can be analysed to gain knowledge about patterns indentifying MOOC students which are likely to fail / withdraw and used to train models capable of predicting their failure / withrawal. One living example is the popular OULAD dataset [^2] which contains data about courses, students and their interactions with Open University's Virtual Learning Environment (VLE) for seven different courses. In the scope of this project we aim to explore the tools and approaches in descriptive analysis, clustering and prediction on the OULAD dataset in order to tackle the MOOC student failure / withdrawal prediction problem.

[^2] : Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).

## Project structure

As work is still in progress the project structure might change.
At this point in time the project is paritionned in 3 jupyter notebooks and one tools directory:

- DescriptiveAnalysis notebook
    - Consists of a very basic descriptive analysis of the OULAD dataset. For now we display the table stucture and their value range for each of the 7 tables of the OULAD dataset.
- Clustering notebook
    - Regroups our experiments on clustering and consensus clustering approaches
- Prediction notebook
    - Regroups our experiments on prediction approaches
- Tools directory
    - Regroups common function definitions used by the jupyter notebooks

## Data cleaning

A detailed description of the OULAD dataset data can be found at https://analyse.kmi.open.ac.uk/open_dataset <br>
After the first basic descriptive analysis we have noticed that 82% of `vle.week_from` `vle.week_to` values are missing.<br>
So we have removed these columns from the vle table.<br>
The `studentInfo.imd_band` column had 3.4% missing values.<br>
We have choosen to replace those missing values by the value "0-100%" indicating the abcence of knowledge on the Index of Multiple Depravation band.<br>
These steps are implemented in the `tools/load_oulad.py` module
Also we had noticed that in the `assessments` table the `date` column for assessments of type `Exam` was always missing. Supposing that the exam date is likely to be at the end of a course module we replaced the missing values by the corresponding `courses.module_presentation_length` value.
The same action was taken for missing `studentRegistration.unregistration` values, supposing that if the students didn't unregistraded themselfes before the end of the course they are considered as unregistrated at the end of the course.

## Data pre-processing

Our first attempt consists in transforming the data in order to be ingestible for most of the clustering / predictive approaches by:
- selecting all infromation available for a choosen course module following the first 2 weeks after the start of the course
    - as student failure / withdrawal prediction is mostly interresting at the beggining or mid-term of a course and not at the end
    - this step is realised by:
        - filterring out information that belongs to other course modules or was recorded after the second week of the course
        - merging all 7 tables by their primary/foreign key relation
    - this step is implemented in the `tools/filter_oulad.py` module `getOneCourse` function
- groupping this infromation by student and reducing each student group to one table line
    - the "reducing each student group to one table line" step is realised by creating features summarizing the available information by student
    - for now the following features were created:
        - 'score_mean': the mean of obtained scores from assessments
        - 'score_sum': the sum of obtained scores from assessments
        - 'date_submitted_mean': the mean of assessments submission dates
        - 'is_banked_sum': the count of banket assessments
        - 'assessment_type_CMA_count': the count of Computer Marked Assessment taken
        - 'assessment_type_TMA_count': the count of Tutor Marked Assessment taken
        - 'date_mean': the mean of the final submission date of the assessments
        - 'weight_mean': the mean weight of the assessment in %
        - 'weight_sum': the sum of weight of the assessment in %
        - 'gender': same as in the studentInfo table
        - 'region': same as in the studentInfo table
        - 'highest_education': same as in the studentInfo table
        - 'imd_band': same as in the studentInfo table
        - 'age_band': same as in the studentInfo table
        - 'num_of_prev_attempts': same as in the studentInfo table
        - 'studied_credits': same as in the studentInfo table
        - 'disability': same as in the studentInfo table
        - 'final_result': same as in the studentInfo table
        - 'date_registration': same as in the studentInfo table
        - 'date_unregistration': same as in the studentInfo table
        - 'sum_click_mean':  the number of times a student interacts with any material during the period
        - 'sum_click_sum': the mean count of times a student interacts with any material during a day
        - 'activity_type_(`activity_type_name`)':  the number of times a student interacts with material of type `activity_type_name` during the period
    - this step is implemented in the `tools/filter_oulad.py` module `restructure` function
- cleaning and encoding the final table
    - after the "reducing each student group to one table line" step, new missing values might be introduced in columns ['weight_mean', 'score_mean', 'date_submitted_mean']
    - in this step we replace the missing values for weight_mean' and 'score_mean' by 0 and date_submitted_mean by -1
    - finally we encode all categorical variables in to numbers.
    - for ordinal categorical variables we manually controll the order of the number assignment to keep the order in place
    - this step is implemented in the `tools/filter_oulad.py` module `cleanAndMap` function

## Setup

### Prerequisites

0. install pip3 packages: <br>
pip3 install
graphviz
jupyterlab
missingno
numpy
matplotlib
pandas
pyamg
scikit-learn
scikit-learn-extra
scipy
seaborn
tensorflow
1. download the oulad dataset from https://analyse.kmi.open.ac.uk/open_dataset
2. extract all files here in the OULAD directory
3. cd FCI
4. git clone https://github.com/slide-lig/plcmpp.git
5. cd plcmpp/src
6. make
7. cd ../../..

### Running the jupyter notebook

8. jupiter notebook
