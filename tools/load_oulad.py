""" Charging the OULAD dataset into memory,
filling studentInfo['imd_band'] NaNs with '0-100%',
removing vle['week_from'], vle['week_to'] as 82% of them are NaNs
This Module is intended to be imported by the jupyter notebooks at
the root directory of the project
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Charging the OULAD dataset into memory
assessments_df         = pd.read_csv('./OULAD/assessments.csv')
courses_df             = pd.read_csv('./OULAD/courses.csv')
studentAssessment_df   = pd.read_csv('./OULAD/studentAssessment.csv')
studentInfo_df         = pd.read_csv('./OULAD/studentInfo.csv')
studentRegistration_df = pd.read_csv('./OULAD/studentRegistration.csv')
studentVle_df          = pd.read_csv('./OULAD/studentVle.csv')
vle_df                 = pd.read_csv('./OULAD/vle.csv')

# some of imd_band are NaN -> replacing them with 0-100%
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='0-100%')
studentInfo_df['imd_band'] = imp.fit_transform(studentInfo_df[['imd_band']])

# 82% of vle.week_from vle.week_to are NaN
# of the remaining 18% rows - in 99.2% of cases vle.week_from is equal to vle.week_to
# with this in mind - we can ignore this data
del vle_df['week_from']
del vle_df['week_to']

# we store references
dataset_dict = {
    'assessments': assessments_df,
    'courses': courses_df,
    'studentAssessment': studentAssessment_df,
    'studentInfo': studentInfo_df,
    'studentRegistration': studentRegistration_df,
    'studentVle': studentVle_df,
    'vle': vle_df
}

