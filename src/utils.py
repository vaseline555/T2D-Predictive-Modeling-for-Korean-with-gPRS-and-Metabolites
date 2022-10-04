import os
import optuna
import json

import numpy as np
import pandas as pd

from collections import defaultdict
from joblib import Parallel, delayed

from sklearn.utils import resample
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, brier_score_loss, log_loss, roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from BorutaShap import BorutaShap
from xgboost import XGBClassifier

def net_benefit(y_test, y_prob, p_min=0.01, p_max=1.00, p_interval=0.01, n_jobs=-1, verbose=False):
    # define workhorse
    def calculate_net_benefit(p):
        y_pred = y_prob >= p
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        n = tn + fp + fn + tp
        net_benefit = (tp / n) - (fp / n) * (p / (1 - p))
        return net_benefit
    
    # calculate net benefits
    net_benefits = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(calculate_net_benefit)(p) for p in np.arange(p_min, p_max, p_interval)
    )
    return net_benefits

def custom_cross_validate(preprocessor, estimator, X, y, cv_scheme, random_state, n_repeats, select_features=False, hyperopt=True, hyperparam_file_name=None, file_name=None, n_jobs=-1, verbose=True):
    # bootrstrap workhorse
    def bootstrap_workhorse(iters, y_test, y_prob):
        y_test, y_prob = resample(y_test, y_prob, random_state=iters)
        
        results = dict()
        # find optimal threshold using G-mean
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # calculate the G-mean for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))

        # locate the index of the largest G-mean
        idx = np.argmax(gmeans)
        results['best_thresholds'] = thresholds[idx]
        
        # get optimal labels
        y_pred = np.where(y_prob >= thresholds[idx], 1, 0)
        
        # keep outputs
        results['y_prob'] = y_prob        
        results['y_pred'] = y_pred
        results['y_test'] = y_test.values
        
        # get scores induced by hard labels
        results['test_accuracy'] = accuracy_score(y_test, y_pred)
        results['test_recall'] = recall_score(y_test, y_pred)
        results['test_precision'] = precision_score(y_test, y_pred)
        results['test_f1'] = f1_score(y_test, y_pred)
        results['test_specificity'] = recall_score(y_test, y_pred, pos_label=0)
        results['test_mcc'] = matthews_corrcoef(y_test, y_pred)

        # get scores induced by predicted probabilities
        results['test_brier'] = brier_score_loss(y_test, y_prob)
        results['test_logloss'] = log_loss(y_test, y_prob)
        results['test_auroc'] = roc_auc_score(y_test, y_prob)
        results['test_auprc'] = average_precision_score(y_test, y_prob)
        results['test_net_benefit'] = net_benefit(y_test, y_prob, n_jobs=n_jobs, verbose=False)
        return results
        
    # define workhorse method
    def cv_workhorse(fold_idx, estimator, train_index, test_index):            
        # training-test split in a fold
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
            
        # preprocess data
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        # FEATURE SELECTION
        if select_features:
            fs_results = {}
            modified_cols = preprocessor.get_feature_names_out()
            target_indices = [i for i, elem in enumerate(modified_cols) if 'candidates' in elem]
            
            X_train_df = pd.DataFrame(X_train, columns=modified_cols)
            X_test_df = pd.DataFrame(X_test, columns=modified_cols)
            
            selector = BorutaShap(importance_measure='gini', classification=True)
            selector.fit(X_train_df.iloc[:, target_indices], y_train, n_trials=100, train_or_test='test', normalize=True, verbose=False, random_state=random_state)
            
            selected_cols = selector.Subset().columns.tolist()
            X_train = np.concatenate((np.delete(X_train, target_indices, axis=1), X_train_df.loc[:, selected_cols]), axis=1)
            X_test = np.concatenate((np.delete(X_test, target_indices, axis=1), X_test_df.loc[:, selected_cols]), axis=1)
            
            # save files
            df_fi = pd.DataFrame(data={'Features': selector.history_x.iloc[1:].columns.values,
                                       'Average Feature Importance': selector.history_x.iloc[1:].mean(axis=0).values,
                                       'Standard Deviation Importance': selector.history_x.iloc[1:].var(axis=0).values})
            decision_mapper = selector.create_mapping_of_features_to_attribute(maps=['Tentative','Rejected','Accepted', 'Shadow'])
            df_fi['Decision'] = df_fi['Features'].map(decision_mapper)
            df_fi = df_fi.sort_values(by='Features', ascending=False)
            fs_results[f'features_{str(fold_idx + 1).zfill(2)}'] = df_fi

        # hyperparameter tuning
        if "RandomForest" in estimator.__class__.__name__ and hyperopt:# and fold_idx % 10 == 0:
            def _rf_objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 500)
                min_samples_split = trial.suggest_int('min_samples_split', 20, 200)
                max_leaf_nodes = int(trial.suggest_int('max_leaf_nodes', 5, 100))
                min_samples_leaf = int(trial.suggest_int('min_samples_leaf', 2, 100))
                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
                oob_score = trial.suggest_categorical('oob_score', [True, False])
                max_features = trial.suggest_categorical('max_features', ['auto', 'log2'])
                class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
                
                rf_clf = RandomForestClassifier(n_estimators=n_estimators,
                                                min_samples_split=min_samples_split,
                                                max_leaf_nodes=max_leaf_nodes,
                                                min_samples_leaf=min_samples_leaf,
                                                criterion=criterion,
                                                oob_score=oob_score,
                                                max_features=max_features,
                                                class_weight=class_weight,
                                                n_jobs=n_jobs,
                                                random_state=random_state)
                """
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.33)
                rf_clf.fit(X_tr, y_tr)
                y_pr = rf_clf.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, y_pr)
                """
                score = cross_validate(rf_clf, X_train, y_train, 
                                       cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=random_state),
                                       scoring='roc_auc',
                                       n_jobs=n_jobs)['test_score']
                return score.mean()
                
            rf_study = optuna.create_study(direction='maximize')
            rf_study.optimize(_rf_objective, n_trials=5)
            rf_trial = rf_study.best_trial
            
            # fit training data
            #X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train, random_state=fold_idx)
            estimator.set_params(**rf_trial.params, n_jobs=n_jobs).fit(X_train, y_train)
            #estimator.set_params(**rf_trial.params, n_jobs=n_jobs).fit(X_tr, y_tr)
            #estimator = CalibratedClassifierCV(base_estimator=estimator, cv='prefit', n_jobs=-1)
            #estimator.fit(X_val, y_val)
        elif "XGB" in estimator.__class__.__name__ and hyperopt: #and fold_idx % 10 == 0:
            def xgb_objective(trial):
                param = {
                    'tree_method': 'gpu_hist',
                    'lambda': trial.suggest_loguniform('lambda', 1e-3, 30.0),
                    'alpha': trial.suggest_loguniform('alpha', 1e-3, 30.0),
                    'colsample_bytree':  trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1.0, 0.1),
                    'subsample': trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1),
                    'learning_rate': trial.suggest_discrete_uniform('learning_rate', 0.01, 0.1, 0.01),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 2, 6),
                    'min_child_weight': trial.suggest_int('min_child_weight', 10, 30),
                    'scale_pos_weight': trial.suggest_int('scale_pos_weight', 10, 30),
                    'eval_metric': 'auc',
                    'verbosity': 1,
                    'n_jobs': n_jobs
                }
                xgb_clf = XGBClassifier(**param)  
                score = cross_validate(xgb_clf, X_train, y_train, 
                                       cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=random_state),
                                       scoring='roc_auc',
                                       n_jobs=n_jobs)['test_score']
                return score.mean() 
            xgb_study = optuna.create_study(direction='maximize')
            xgb_study.optimize(xgb_objective, n_trials=5)
            xgb_trial = xgb_study.best_trial
            
            # fit training data
            #X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train, random_state=fold_idx)
            estimator.set_params(**xgb_trial.params, n_jobs=n_jobs).fit(X_train, y_train)
            #estimator.set_params(**xgb_trial.params, n_jobs=n_jobs).fit(X_tr, y_tr)
            #estimator = CalibratedClassifierCV(base_estimator=estimator, cv='prefit', n_jobs=-1)
            #estimator.fit(X_val, y_val)
        else:
            #X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train, random_state=fold_idx)
            estimator.fit(X_train, y_train)
            #estimator.fit(X_tr, y_tr)
            #estimator = CalibratedClassifierCV(base_estimator=estimator, cv='prefit', n_jobs=-1)
            #estimator.fit(X_val, y_val)
            
        # predict class probabilities
        y_prob = estimator.predict_proba(X_test)[:, 1]
            
        # bootstrap validation results
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(bootstrap_workhorse)(iters, y_test, y_prob) for iters in range(n_repeats)
        )
        
        # merge resulting scores
        result_dict = defaultdict(list)
        for d in results:
            for k, v in d.items():
                result_dict[k].append(v)
        if select_features:
            result_dict = dict(result_dict)
            result_dict.update(fs_results)
            return result_dict
        return result_dict
    
    # run workhorse
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(cv_workhorse)(fold_idx, estimator, train_index, test_index) for fold_idx, (train_index, test_index) in enumerate(cv_scheme.split(X, y))
    )
    
    # merge resulting scores
    result_dict = defaultdict(list)
    for d in results:
        for k, v in d.items():
            if 'features' in k:
                result_dict[k].append(v)
            else:
                result_dict[k].extend(v)           
    return result_dict

def manipulated_feature_selection_report(result_dict, file_name):
    fi_dataset = []
    for name, elements in result_dict.items():
        if 'features' in name:
            fi_dataset.append(elements[0])

    fi_df_all = pd.concat(fi_dataset)
    fi_df_all_new = fi_df_all.groupby('Features')['Average Feature Importance', 'Standard Deviation Importance'].mean()
    fi_df_all_new.loc[:, 'Standard Deviation Importance'] = fi_df_all_new.loc[:, 'Standard Deviation Importance'].pow(0.5)

    decision_df = fi_df_all.groupby('Features')['Decision'].agg(lambda col: ','.join(col))
    fi_df_all_new['Accepted'] = decision_df.str.count('Accepted')
    fi_df_all_new['Rejected'] = decision_df.str.count('Rejected')
    fi_df_all_new['Teantative'] = decision_df.str.count('Teantative')
    
    fi_df_all_new = fi_df_all_new.reset_index()
    fi_df_all_new['Features'] = fi_df_all_new['Features'].str.replace('candidates__', '')
    fi_df_all_new = fi_df_all_new[~fi_df_all_new['Features'].str.contains("Shadow")]
    
    fi_df_all_new.to_csv(file_name, index=False)
    return fi_df_all_new
