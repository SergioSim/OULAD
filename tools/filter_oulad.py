""" Filtering out one specific course and merging all tables into one
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def getOneCourse(dataset_dict, code_module, code_presentation):
    '''
    Merge all tables by their primary keys for a single course,
    replace NaNs of missing Exam/unregistration dates with module_presentation_length
    '''
    def filterByCode(table):
        '''returns a boolean pd.Series of same table length'''
        return  (table['code_module'] == code_module) & \
                (table['code_presentation'] == code_presentation)
    
    course = dataset_dict['courses']
    course = course[filterByCode(course)]
    module_presentation_length = course['module_presentation_length'].values[0]
    
    # studentAssessment -> complete studentAssessments with assessments info
    assessments = dataset_dict['assessments']
    assessments.loc[ (assessments['assessment_type'] == 'Exam') & \
        filterByCode(assessments), \
        'date'] = module_presentation_length
    assessments = assessments[filterByCode(assessments)]
    studentAssessment = pd.merge(dataset_dict['studentAssessment'], assessments, \
                                 how='inner', on='id_assessment')

    # studentInfo -> complete studentInfo with studentRegistration
    studentRegistration = dataset_dict['studentRegistration']
    studentRegistration.loc[studentRegistration['date_unregistration'].isna() & \
        filterByCode(studentRegistration), \
        'date_unregistration'] = module_presentation_length
    studentRegistration = studentRegistration[ \
                           filterByCode(studentRegistration)]
    studentInfo = pd.merge(dataset_dict['studentInfo'], studentRegistration, \
                           how='inner', on=['id_student', 'code_module', 'code_presentation'])

    # studentVle = complete vle with studentVle
    vle = dataset_dict['vle']
    vle = vle[filterByCode(vle)]
    studentVle = pd.merge(dataset_dict['studentVle'], vle, \
                          how='inner', on=['id_site', 'code_module', 'code_presentation'])
    del studentVle['id_site']
    
    # complete with studentVle with studentInfo, complete studentAssessment with studentInfo
    # then append enhanced studentVle with studentAssessment, sort by date
    studentVleInfo = pd.merge(studentVle, studentInfo, \
                              how='inner', on=['id_student', 'code_module', 'code_presentation'])
    studentAssessmentInfo = pd.merge(studentAssessment, studentInfo, \
                              how='inner', on=['id_student', 'code_presentation', 'code_module'])
    
    combined_df = studentAssessmentInfo.append(studentVleInfo)
    del combined_df['code_module']
    del combined_df['code_presentation']
    del combined_df['id_assessment']
    combined_df = combined_df.sort_values(by=['date'])
    return combined_df

"""
Restructuring the oneCourse table:
oneCourse table might be good for algorithms that take into account the sequence / time
to make a first POC we will simplify drasticaly
    remove every thing that is beyond day 14
    aggregate each student sequence to make for each student contain only one row
        date_submitted (ignore NaNs) -> compute the mean
        same with date, score
        is_banked (ignore NaNs) -> compute the sum (= how many assessments were banked)
        same with score
        assessment_type -> create 2 variables:
            sumTMA -> sum of TMAs
            sumCMA -> sum of CMAs
        for each activity type make a variable and take the sum of sum_click
"""

def restructure(oneCourse, days):
    '''
    aggregate each student sequence to make for each student contain only one row,
    keeping only the data of the first two weeks
    '''
    # prediction is only then interresting when it's the begining of the course - not the end!
    first14Days_oneCourse = oneCourse[oneCourse['date'] <= days]
    # remove those who unregistered before the begining as we can't do anything for them
    first14Days_oneCourse = first14Days_oneCourse[first14Days_oneCourse['date_unregistration'] \
                                                  > 0]
    # list of unique activity_types to create features
    activity_types_df = first14Days_oneCourse['activity_type'].unique()
    # remove NaN activity type
    activity_types_df = [x for x in activity_types_df if type(x) == str]
    # we want one student per line (the easy way)
    final_df = first14Days_oneCourse.groupby('id_student').agg({
        'score': [np.mean, np.sum],
        'date_submitted': [np.mean],
        'is_banked': [np.sum],
        'assessment_type': [('CMA_count', lambda x: x.values[x.values == 'CMA'].size), \
                            ('TMA_count', lambda x: x.values[x.values == 'TMA'].size)],
        'date':[np.mean],
        'weight': [np.mean, np.sum],
        'gender': [('first', lambda x: x.values[0])],
        'region': [('first', lambda x: x.values[0])],
        'highest_education': [('first', lambda x: x.values[0])],
        'imd_band': [('first', lambda x: x.values[0])],
        'age_band': [('first', lambda x: x.values[0])],
        'num_of_prev_attempts': [('first', lambda x: x.values[0])],
        'studied_credits': [('first', lambda x: x.values[0])],
        'disability': [('first', lambda x: x.values[0])],
        'final_result': [('first', lambda x: x.values[0])],
        'date_registration': [('first', lambda x: x.values[0])],
        'date_unregistration': [('first', lambda x: x.values[0])],
        'sum_click': [np.mean, np.sum],
        'activity_type': [('list', lambda x: [x.values[x.values == activity].size \
                                              for activity in activity_types_df])]
    })
    # keeping only one level of columns names
    final_df.columns = ["_".join(x) for x in final_df.columns.ravel()]
    custom_columns = ['activity_type_' + str(x) for x in activity_types_df]
    # splitting & concatenating the created features (there should be a better way to do it...)
    final_df = final_df.join(pd.DataFrame(final_df.activity_type_list.values.tolist(), \
                                          columns=custom_columns, index=final_df.index))
    del final_df['activity_type_list']
    return final_df


def cleanAndMap(final_df, encode=True):
    '''
    replace NaNs intoduced by new features and if encode=True -> map 
    catogorical columns to numbers
    returning the encoder object to decode the labels later on
    '''
    # replacing NaNs
    final_df.loc[final_df['weight_mean'].isna(), 'weight_mean'] = 0
    final_df.loc[final_df['score_mean'].isna(), 'score_mean'] = 0
    final_df.loc[final_df['date_submitted_mean'].isna(), 'date_submitted_mean'] = -1

    if not encode:
        return
        
    # replacing categorical features with numbers
    # manualy handling order for ordinal features
    categorical_columns = ['gender_first', 'disability_first', 'region_first', 'age_band_first']
    ordinal_columns = {
        'highest_education_first': {
            'No Formal quals': 0,
            'Lower Than A Level': 1,
            'A Level or Equivalent': 2,
            'HE Qualification': 3,
            'Post Graduate Qualification': 4
        },
        'imd_band_first': {
            '0-100%': 0, # may be we could put this in 50 ?
            '0-10%': 5,
            '10-20': 15,
            '20-30%': 25,
            '30-40%': 35,
            '40-50%': 45,
            '50-60%': 55,
            '60-70%': 65,
            '70-80%': 75,
            '80-90%': 85,
            '90-100%': 95
        },
        'final_result_first': {
            'Withdrawn': 0,
            'Fail': 1,
            'Pass': 2,
            'Distinction': 3,
        }
    }
    categorical_encoders = {col: LabelEncoder() for col in categorical_columns }
    for col in categorical_columns:
        final_df.loc[:, col] = categorical_encoders[col].fit_transform(final_df[col])
    for col in ordinal_columns:
        final_df.loc[:, col] = final_df.loc[:, col].map(ordinal_columns[col])
        
    return categorical_encoders



