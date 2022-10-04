import os
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils import *
from src.plot import *
from src.compare import *


if __name__ == "__main__":
    # get configurations
    parser = argparse.ArgumentParser(description='Type2 Diabetes Risk Prediction with genome-wide Polygenic Risk Score and Serum Metabolites')
    parser.add_argument('--data_path', type=str, default='./data', help='path to read data')
    parser.add_argument('--result_path', type=str, default='./result', help='path to save resulting dataframes')
    parser.add_argument('--plot_path', type=str, default='./plot', help='path to save resulting plots')
    parser.add_argument('--hyperparam_path', type=str, default='./hyperparam', help='path to save resulting hyperparameters')
    parser.add_argument('-k', '--n_splits', type=int, default=10, help='number of folds to use for cross validation')
    parser.add_argument('-r', '--n_repeats', type=int, default=200, help='number of repetition to use for cross validation')
    parser.add_argument('-s', '--random_state', type=int, default=1001, help='random seed for reproducibility')
    
    # parse arguments
    args = parser.parse_args()
    
    # sanity check
    args.result_path = os.path.join(args.result_path, str(args.random_state))
    args.plot_path = os.path.join(args.plot_path, str(args.random_state))
    
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.plot_path, exist_ok=True)
    #os.makedirs(args.hyperparam_path, exist_ok=True)
    
    # empty container for comparison
    comparisons = {}
    
    # load all data
    dataset = {}
    for file in sorted(os.listdir(args.data_path)):
        dataset[file[:-4]] = pd.read_csv(os.path.join(args.data_path, file), index_col=0)
    
    # wrangle data
    for exp_name, data in tqdm(dataset.items(), desc='Experiemnt Progress'):
        print(f"[INFO] Experiment [{exp_name.upper()}]")
        # set flag for feature selection
        if 'type4' in exp_name:
            select_features = True
        else:
            select_features = False
        
        # split data
        X, y = data.drop(["DM_YN"], axis=1), data["DM_YN"]
        cols = X.columns.tolist()

        cat_vars = ["SEX", "HT", "FMDMREL", "SMOKE"]
        if select_features:
            cand_vars = cols[12:-1]
        else:
            cand_vars = []
        num_vars = list(set(cols) - set(cat_vars) - set(cand_vars))
    
        # define categorical pipeline
        cat_pipe = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        # define numerical pipeline
        num_pipe = Pipeline([
            ('scaler', StandardScaler())
        ])

        # combine categorical and numerical pipelines as a preprocessor module
        preprocessor = ColumnTransformer([
            ('cat', cat_pipe, cat_vars),
            ('num', num_pipe, num_vars),
            ('candidates', num_pipe, cand_vars)
        ])

        # prepare estimators
        estimators = {
            'Logistic Regression': LogisticRegression(penalty='none', max_iter=1e8, n_jobs=-1),
            #'Logistic Regression (Lasso)': LogisticRegression(penalty='l1', solver='saga', max_iter=1e5, n_jobs=-1),
            #'Logistic Regression (Ridge)': LogisticRegression(penalty='l2', max_iter=1e5, n_jobs=-1),
            #'Logistic REgression (ElasticNet)': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1e5, n_jobs=-1),
            'Random Forest': RandomForestClassifier(random_state=args.random_state, n_jobs=-1),
            #'Extreme Gradient Boosting': XGBClassifier(random_state=args.random_state, n_jobs=-1)
        }
        
        # conduct experiments for each estimator 
        rskf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=1, random_state=args.random_state)
        for estimator_name, estimator in tqdm(estimators.items(), desc='Estimator Progress'):
            print(f"[INFO] Estimator [{estimator_name.upper()}]")
            result_dict = custom_cross_validate(preprocessor, estimator, X, y, n_repeats=args.n_repeats,
                                                cv_scheme=rskf, random_state=args.random_state, select_features=select_features,
                                                hyperopt=True,
                                                hyperparam_file_name=os.path.join(args.hyperparam_path, f'{exp_name}_{estimator_name}'),
                                                file_name=os.path.join(args.result_path, f'{exp_name}_{estimator_name}'))

            # collect resulting metrics
            metrics = ['best_thresholds', 'test_accuracy', 'test_recall', 'test_precision', 'test_f1', 'test_specificity', 'test_mcc', 'test_brier', 'test_logloss', 'test_auroc', 'test_auprc']
            df_result = pd.DataFrame({metric: result_dict[metric] for metric in metrics}).T
            df_result['mean'] = df_result.mean(axis=1)
            df_result['std'] = df_result.std(axis=1)
            df_result.loc[:, ['mean', 'std']].to_csv(os.path.join(args.result_path, f'{exp_name}_{estimator_name}_metrics.csv'))
            
            # collect raw outputs
            raw_outputs = ['y_test', 'y_prob', 'y_pred']
            compare_df = pd.DataFrame({raw_output: result_dict[raw_output] for raw_output in raw_outputs})
            #compare_df.to_csv(os.path.join(args.result_path, f'{exp_name}_{estimator_name}_comparison.csv'))
            comparisons[f'{exp_name}_{estimator_name}'] = compare_df
            
            if select_features:
                # save results on selected features
                fi_df_all_new = manipulated_feature_selection_report(result_dict, os.path.join(args.result_path, f'{exp_name}_{estimator_name}_feature_selection.csv'))
                
                # plot boxplots on selected features
                plot_feature_importances(os.path.join(args.plot_path, f'{exp_name}_{estimator_name}_fi.png'), fi_df_all_new, args.n_splits)
                
            # plot net benefits (decision curve analysis)
            plot_decision_curve_analysis(os.path.join(args.plot_path, f'{exp_name}_{estimator_name}_dca.png'), estimator_name, np.concatenate(result_dict['y_test']), result_dict['test_net_benefit'])

            # plot roc curve
            plot_roc_curve(os.path.join(args.plot_path, f'{exp_name}_{estimator_name}_roc.png'), estimator_name, result_dict["y_test"], result_dict["y_prob"], result_dict["test_auroc"])

            # plot det curve
            plot_det_curve(os.path.join(args.plot_path, f'{exp_name}_{estimator_name}_det.png'), estimator_name, result_dict["y_test"], result_dict["y_prob"])
            
            # plot prc curve
            plot_prc_curve(os.path.join(args.plot_path, f'{exp_name}_{estimator_name}_prc.png'), estimator_name, result_dict["y_test"], result_dict["y_prob"], result_dict["test_auprc"])

            # plot calibration curve
            plot_calibration_curve(os.path.join(args.plot_path, f'{exp_name}_{estimator_name}_calib.png'), estimator_name, result_dict["y_test"], result_dict["y_prob"])
    
    print("[INFO] all individual experiments are done...!\n[INFO] start comparison between experiments...!")
    # compare results
    compare_lists = list(comparisons.keys())
    compare_lists.sort(key=lambda x: x.split()[-1])

    # start comparison
    for exp1, exp2 in zip(compare_lists, compare_lists[1:]):  
        y1_test = comparisons[exp1]['y_test'].values.reshape(-1, 1)
        y2_test = comparisons[exp2]['y_test'].values.reshape(-1, 1)
        y1_prob = comparisons[exp1]['y_prob'].values.reshape(-1, 1)
        y2_prob = comparisons[exp2]['y_prob'].values.reshape(-1, 1)
        y1_pred = comparisons[exp1]['y_pred'].values.reshape(-1, 1)
        y2_pred = comparisons[exp2]['y_pred'].values.reshape(-1, 1)      
        
        results = defaultdict(list)
        for y1_te, y1_pr, y1_p, y2_te, y2_pr, y2_p in tqdm(zip(y1_test, y1_prob, y1_pred, y2_test, y2_prob, y2_pred)):
            df = pd.DataFrame({'y1_test': [y1_te[0]],
                               'y2_test': [y2_te[0]],
                               'y1_prob': [y1_pr[0]],
                               'y2_prob': [y2_pr[0]],
                               'y1_pred': [y1_p[0]],
                               'y2_pred': [y2_p[0]]})
            df = df.apply(pd.Series.explode)
            
            result_nri = NRI(df)
            result_cnri = cNRI(df)
            result_idi = IDI(df)

            results['NRI_cv'].append(result_nri[0])
            results['NRI_z'].append(result_nri[1])

            results['cNRI_cv'].append(result_cnri[0])
            results['DD_cv'].append(result_cnri[1])
            results['SS_cv'].append(result_cnri[2])
            results['cNRI_z'].append(result_cnri[3])

            results['IDI_cv'].append(result_idi[0])
            results['IDI_z'].append(result_idi[1])

        comparison_df = pd.DataFrame({"NRI_mean": np.nanmean(results['NRI_cv']),
                                      "NRI_std": np.nanstd(results['NRI_cv']),
                                      "p-value (NRI)": [1 - stats.binom.cdf(np.nanmean(results['NRI_z']), len(results['NRI_z']), 0)],
                                      "cNRI_mean": np.nanmean(results['cNRI_cv']),
                                      "cNRI_std": np.nanstd(results['cNRI_cv']),
                                      "p_value (cNRI)": [1 - stats.binom.cdf(np.nanmean(results['cNRI_z']), len(results['NRI_z']), 0)],
                                      "IDI_mean": np.nanmean(results['IDI_cv']),
                                      "IDI_std": np.nanstd(results['IDI_cv']),
                                      "p_value (IDI)": [1 - stats.norm(0, 1).cdf(np.nanmean(results['IDI_z']))],
                                      "p_value (DeLong AUC)": delong_roc_test(
                                          np.hstack(y1_test.tolist()).ravel(), 
                                          np.hstack(y1_prob.tolist()).ravel(), 
                                          np.hstack(y2_prob.tolist()).ravel())[0]}, index=['value'])
        comparison_df.to_csv(f'RESULT_{args.random_state}_comparison_{exp1} & {exp2}.csv')
        
    # aggregate all metrics
    result_df = []
    for file in sorted(os.listdir(args.result_path)):
        if 'metrics' in file:
            df = pd.read_csv(os.path.join(args.result_path, file), index_col=0)
            df = df.T
            df["model"] = file[6:-12]
            result_df.append(df)
    final_result_df = pd.concat(result_df).reset_index().set_index('model').rename(columns={'index': 'statistics'})
    final_result_df.to_csv(f'RESULT_{args.random_state}_aggregated_metrics.csv')