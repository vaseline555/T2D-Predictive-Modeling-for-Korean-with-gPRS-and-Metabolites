import numpy as np
import scipy.stats as stats
                                                  
def NRI(df):
    true_event = (df['y1_test'] == 1).sum()
    true_nonevent = (df['y1_test'] == 0).sum()
    
    # event
    dpn = len(df[(df['y1_test'] == 1) & (df['y1_pred'] == 1) & (df['y2_pred'] == 0)]) # 5
    dnp = len(df[(df['y1_test'] == 1) & (df['y1_pred'] == 0) & (df['y2_pred'] == 1)]) # 2
    
    # non-event
    npn = len(df[(df['y1_test'] == 0) & (df['y1_pred'] == 1) & (df['y2_pred'] == 0)]) # 7
    nnp = len(df[(df['y1_test'] == 0) & (df['y1_pred'] == 0) & (df['y2_pred'] == 1)]) # 4

    NRI = (dnp - dpn) / true_event + (npn - nnp) / true_nonevent
    NRI_z = NRI / np.sqrt((dnp + dpn) / true_event + (npn + nnp) / true_nonevent)
    NRI_Z = np.nan_to_num(NRI_z, copy=False)
    return NRI, NRI_z


def cNRI(df):
    DH = len(df[(df['y1_test'] == 1) & (df['y2_prob'] > df['y1_prob'])])
    DL = len(df[(df['y1_test'] == 1) & (df['y2_prob'] < df['y1_prob'])])
    
    SH = len(df[(df['y1_test'] == 0) & (df['y2_prob'] > df['y1_prob'])])
    SL = len(df[(df['y1_test'] == 0) & (df['y2_prob'] < df['y1_prob'])])

    DD = (DH - DL) / (DH + DL)
    SS = (SL - SH) / (SL + SH)
    pe_up = DH / (DH + DL)
    pne_up = SL / (SL + SH)
    
    cNRI = (DH - DL) / (DH + DL) + (SL - SH) / (SL + SH)
    cNRI_z = cNRI / (np.sqrt(4 * (pe_up * (1 - pe_up) / (DH + DL) + pne_up * (1 - pne_up) / (SH + SL))))
    return cNRI, DD, SS, cNRI_z

def IDI(df):
    mp_2_prob = df.loc[df['y1_test'] == 1, 'y2_prob']
    mp_1_prob = df.loc[df['y1_test'] == 1, 'y1_prob']
    
    mn_2_prob = df.loc[df['y1_test'] == 0, 'y2_prob']
    mn_1_prob = df.loc[df['y1_test'] == 0, 'y1_prob']
    
    mp_2 = mp_2_prob.mean()
    mp_1 = mp_1_prob.mean()
    mn_2 = mn_2_prob.mean()
    mn_1 = mn_1_prob.mean()
    
    se_p = (mp_2_prob - mp_1_prob).std() / np.sqrt(len(mp_2_prob))
    se_n = (mn_2_prob - mn_1_prob).std() / np.sqrt(len(mp_2_prob))
    
    IDI = (mp_2 - mp_1) + (mn_1 - mn_2)
    IDI_z = IDI / np.sqrt((se_p)**2 + (se_n)**2)
    return IDI, IDI_z

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return 10**calc_pvalue(aucs, delongcov)