"""Custom clustor quality measure"""
import pandas as pd

def customClusteringScore(clustAlgo_labels, testY):
    """ returns our clustering quality score
    For each cluster we compute a sub_score: 
    = abs((Pass + Distinction) - (Fail + Withdrawn)) / cluster_size
    The returned score corresponds to the sum of all sub_scores
    """
    compare_df = pd.concat([testY, clustAlgo_labels], axis=1)
    # bring back the final_result labels (not really needed but makes the code clearer)
    decode_map = {
            0: 'Withdrawn',
            1: 'Fail',
            2: 'Pass',
            3: 'Distinction'
    }
    compare_df.loc[:,'final_result_first'] = \
        compare_df.loc[:,'final_result_first'].map(decode_map)
    compare_df = compare_df.groupby([clustAlgo_labels.name, 'final_result_first']).size()
    # the adjusted_rand_score measure is good for measuring the quality of the cluster
    #   in general
    # print('adjusted_rand_score: ', adjusted_rand_score(testY, kmeans_labels))

    # but we want to measure how good the clustering algorithm is able to separate 
    #  Distinction/Pass from Fail/Withdrawn:
    # the measure should be close or equal to 0 if both groups are almost equal or equal in size
    # should be close or equal to 1 if one of the groups is small compared to the other
    # should be between (0,1) if g1 > g2 / g1 < g2
    # proposed measure: score = abs(g1 - g2) / g1 + g2
    # we want to take into account the size of the groups too
    # we prefer 'big' groups with a good score rather than small groups with exellent scores
    # for that we could normalize the group score : score * group-size / total size
    # a perfect clustering would be one that has only 2 groups each of them of score 1
    # small groups would have a little weight compared to others but can sum up to 1
    # for now let try not to penalize by group count...
    nb_clusters = clustAlgo_labels.nunique()
    scores = []
    for cluster in range(0,nb_clusters):
        failWihdrawn = compare_df[cluster]['Fail'] \
                            if 'Fail' in compare_df[cluster].index else 0
        failWihdrawn += compare_df[cluster]['Withdrawn'] \
                            if 'Withdrawn' in compare_df[cluster].index else 0
        PassDistinction = compare_df[cluster]['Pass'] \
                            if 'Pass' in compare_df[cluster].index else 0
        PassDistinction += compare_df[cluster]['Distinction'] \
                            if 'Distinction' in compare_df[cluster].index else 0
        scores.append(\
            (abs(PassDistinction - failWihdrawn) / (PassDistinction + failWihdrawn)) *\
                      (PassDistinction + failWihdrawn) / len(testY))
    return sum(scores)