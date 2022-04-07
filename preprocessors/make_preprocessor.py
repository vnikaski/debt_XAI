from typing import List
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA, TruncatedSVD

def make_preprocessor(processes: List[str], os_rate: float, us_rate: float, cur_ratio: float, n_components: int) -> List[tuple]:
    """
    :param processes: list of processes from args
    :param os_rate: rate of oversmapling (1 = None, 2 = twice as many)
    :param us_rate: rate of undersampling (1 = None, 2 = half of all) WARNING: us is done AFTER os and takes its changes
     into consideration
    :param cur_ratio: current ratio of data, WARNING cur_ratio changes after each resampling
    :return: List of tuples (str, Any) consisting of name of a preprocess and the preprocess
    """

    if processes is None:
        return []

    preps = []
    if 'ROS' in processes:
        preps.append(('ROS', RandomOverSampler(sampling_strategy=cur_ratio * os_rate, random_state=0)))
        cur_ratio = cur_ratio * os_rate
    if 'SMOTE' in processes:
        preps.append(('SMOTE', SMOTE(sampling_strategy=cur_ratio * os_rate, random_state=0)))
        cur_ratio = cur_ratio * os_rate
    if 'RUS' in processes:
        preps.append(('RUS', RandomUnderSampler(sampling_strategy=cur_ratio * us_rate, random_state=0)))
    if 'normalize' in processes: # not sure if normalizer would work in the pipeline
        preps.append(('Normalizer', Normalizer(norm='l2')))
    if 'pca' in processes:
        preps.append(('PCA', PCA(n_components=n_components, random_state=0)))
    if 'svd' in processes:
        preps.append(('SVD', TruncatedSVD(n_components=n_components, random_state=0)))

    return preps
