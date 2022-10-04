import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import to_hex
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, det_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def plot_decision_curve_analysis(file_name, model_name, y_test, net_benefits, p_min=0.01, p_max=1.00, epsilon=0.01):
    num_pos = np.sum(y_test)
    num_neg = y_test.shape[0] - num_pos
    num_total = y_test.shape[0]
    
    # define figure size
    plt.figure(figsize=(15, 10), dpi=500)

    # plot estimated net benefit
    ps = np.arange(p_min, p_max, epsilon)
    for idx, net_benefit in enumerate(net_benefits):
        plt.plot(ps, net_benefit, alpha=0.01, color='gray')
        plt.ioff()
        
    # plot average line
    y, error = tolerant_mean(net_benefits)
    plt.plot(ps, y, color='green', label='Prediction Model (Averaged)')
    plt.fill_between(ps, y - error, y + error, color='green', alpha=0.1)

    # plot 'treat none'
    plt.hlines(y=0, xmin=0, xmax=1, label='Treat None', linestyles='dotted', color='red')

    # plot 'treat all'
    net_benefits_all = []
    for p in ps:
        net_benefit_all = (num_pos / num_total) - (num_neg / num_total) * (p / (1 - p))
        net_benefits_all.append(net_benefit_all)
    plt.plot(ps, net_benefits_all, label='Treat All', linestyle='dashed', color='black')

    # plot
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(f'Decision Curve Analysis ({model_name})')
    plt.ylim([-0.1, 0.30])
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    axes = plt.gca()
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
    plt.legend()
    plt.savefig(file_name)
    plt.close('all')
    
    
def plot_roc_curve(file_name, model_name, y_test, y_prob, auroc):
    # define figure size
    plt.figure(figsize=(15, 10), dpi=500)
    
    # plot roc curves
    fprs, tprs, aurocs = [], [], []
    for idx, (y, p, a) in enumerate(zip(y_test, y_prob, auroc)):
        fpr, tpr, _ = roc_curve(y, p)
        fprs.append(fpr); tprs.append(tpr); aurocs.append(a)
        plt.plot(fpr, tpr, alpha=0.01, color='gray')
        plt.ioff()
    plt.plot([0, 1], [0, 1], linestyle='dashed', color='black')
    
    # plot average line
    fpr_mean, _ = tolerant_mean(fprs)
    tpr_mean, _ = tolerant_mean(tprs)
    plt.plot(sorted(fpr_mean), sorted(tpr_mean), color='green', lw=3, label=f'Average ROC (AUROC={np.mean(aurocs):.4f})')
    
    # plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve ({model_name})')
    plt.ylim([-0.05, 1.05])
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    axes = plt.gca()
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
    plt.legend()
    plt.savefig(file_name)
    plt.close('all')
    
    
def plot_det_curve(file_name, model_name, y_test, y_prob):
    # define figure size
    plt.figure(figsize=(15, 10), dpi=500)
    
    # plot det curves
    fprs, fnrs = [], []
    for idx, (y, p) in enumerate(zip(y_test, y_prob)):
        fpr, fnr, _ = det_curve(y, p)
        fprs.append(fpr); fnrs.append(fnr)
        plt.plot(fpr, fnr, alpha=0.01, color='gray')
        plt.ioff()

    # plot average line
    fpr_mean, _ = tolerant_mean(fprs)
    fnr_mean, _ = tolerant_mean(fnrs)

    plt.plot(sorted(fpr_mean)[::-1], sorted(fnr_mean), color='green', lw=3, label=f'Average DET Curve')
    
    # plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title(f'Detection Error Tradeoff Curve ({model_name})')
    plt.ylim([-0.05, 1.05])
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    axes = plt.gca()
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
    plt.legend()
    plt.savefig(file_name)    
    plt.close('all')
    
    
def plot_prc_curve(file_name, model_name, y_test, y_prob, auprc):
    # define figure size
    plt.figure(figsize=(15, 10), dpi=500)
    
    # plot roc curves
    prs, recs, auprcs = [], [], []
    for idx, (y, p, a) in enumerate(zip(y_test, y_prob, auprc)):
        pr, rec, _ =  precision_recall_curve(y, p)
        prs.append(pr); recs.append(rec); auprcs.append(a)
        plt.plot(rec, pr, alpha=0.01, color='gray')
        plt.ioff()
    
    # plot average line
    pr_mean, _ = tolerant_mean(prs)
    rec_mean, _ = tolerant_mean(recs)

    plt.plot(sorted(rec_mean)[::-1], sorted(pr_mean), lw=3, color='green', label=f'Average PRC (AUPRC={np.mean(auprcs):.4f})')
    
    # plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision Recall Curve ({model_name})')
    plt.ylim([-0.05, 1.05])
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    axes = plt.gca()
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
    plt.legend()
    plt.savefig(file_name)
    plt.close('all')
    
    
def plot_calibration_curve(file_name, model_name, y_test, y_prob):
    # define figure size
    plt.figure(figsize=(15, 10), dpi=500)
    
    # plot roc curves
    prob_trues, prob_preds = [], []
    for idx, (y, p) in enumerate(zip(y_test, y_prob)):
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
        prob_trues.append(prob_true); prob_preds.append(prob_pred)
        plt.plot(prob_pred, prob_true, alpha=0.01, color='gray')
        plt.ioff()
    plt.plot([0, 1], [0, 1], linestyle='dashed', color='black')
    
    # plot average line
    prob_true_mean, _ = tolerant_mean(prob_trues)
    prob_pred_mean, _ = tolerant_mean(prob_preds)

    plt.plot(sorted(prob_pred_mean), sorted(prob_true_mean), lw=3, color='green', marker='o', label='Average Calibration Curve')
    
    # plot
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Probability')
    plt.title(f'Calibration Curve ({model_name})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    axes = plt.gca()
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)
    plt.legend()
    plt.savefig(file_name)
    plt.close('all')

def plot_feature_importances(file_name, df, num_colors):
    # sort dataframe
    df = df.sort_values(by='Average Feature Importance', ascending=False).reset_index()
    
    # cmap
    cmap = plt.cm.get_cmap('hsv', num_colors)
    
    # define figure size
    plt.figure(figsize=(30, 10), dpi=500)
    xs = df['Features']
    ys = df['Average Feature Importance']
    es = df['Standard Deviation Importance']
    cs = df['Accepted'].apply(lambda i: cmap(i))
    
    # plot
    plt.xlabel('Predictors')
    plt.ylabel('AVerage Z-Scores')
    plt.title('Feature Importances')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.xticks(rotation=90, fontsize=10)
    
    axes = plt.gca()
    axes.xaxis.label.set_size(15)
    axes.yaxis.label.set_size(15)

    plt.scatter(xs, ys, c=cs, s=50, zorder=3)
    plt.errorbar(xs, ys, es, zorder=0, capsize=0.8, ecolor='k')
    plt.savefig(file_name)
    plt.close('all')