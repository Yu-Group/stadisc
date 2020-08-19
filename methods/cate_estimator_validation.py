import os
import joblib
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

from methods.data_processing import prepare_df, separate_vars
from methods.causal_functions import (get_subgroup_CATE, get_subgroup_t_statistic,
                                      get_subgroup_CATE_std, get_Neyman_ATE)
from methods.cate_estimator_wrappers import (SLearnerWrapper, TLearnerWrapper,
                                             XLearnerWrapper, RLearnerWrapper,
                                             CausalTreeWrapper, CausalForestWrapper,
                                             EnsembleCATEEstimatorWrapper, 
                                             combine_estimator_versions)

#============================================================#
##### 1. Functions for tuning and fitting the estimators #####
#============================================================#

def make_estimator_library(X, t, y, cv, base_learners, param_grids = None,
                           tuned_params = None, n_iter = 200, verbose = 0):
    """
    Make a library of CATE estimators in the form of a dictionary. If 
    hyperparameters are given, return the library of CATE estimators with those 
    hyperparameters. If not, then first tune the hyperparameters before 
    returning the library.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
        Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
        Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    base_learners: dict of sklearn estimators
        Dictionary of sklearn estimators to use as base learners in the 
        meta-learners.
    param_grids: mapping of mappings of string to any
        Dictionary of dictionaries of grids for hyperparameter searches.
    tuned_params: mapping of mappings of string to any
        Dictionary of hyperparameter dictionaries.
    n_iter: int
        Number of iterations for RandomizedSearchCV used to tune parameters
    verbose: int
        verbose argument for RandomizedSearchCV
    
    Returns
    -------
    library: mapping of string to CATEEstimatorWrapper
       A library of tuned CATE estimators
    """
    
    # User must supply either param_grids to tune the CATE estimators, or
    # tuned_params to set the hyperparameters of the CATE estimators
    assert param_grids is not None or tuned_params is not None
    if param_grids is None:
        param_grids = {name: {} for name in base_learners.keys()}
        tune = False
    else:
        tune = True
    
    # Add S-learner estimators
    s_learners = {}
    for name in ["rf", "xgb"]:
        s_learners["s_" + name] = SLearnerWrapper(X, t, y, cv, 
                                      base_learner = base_learners[name],
                                      param_grid = param_grids[name])
        if tune:
            print("Tuning " + "s_" + name)
            s_learners["s_" + name].tune_params(n_iter)
        
    # Add T-learner estimators
    t_learners = {}
    for name, base_learner in base_learners.items():
        t_learners["t_" + name] = TLearnerWrapper(X, t, y, cv, 
                                      base_learner = base_learner, 
                                      param_grid = param_grids[name])
        if tune:
            print("Tuning " + "t_" + name)
            t_learners["t_" + name].tune_params(n_iter)
            
    # Add X-learner estimators
    x_learners = {}
    effect_learner = "lasso"
    for name, base_learner in base_learners.items():
        # Initialize the XLearnerWrapper object to have the same params_treat 
        # and params_control as that of the TLearnerWrapper with the same 
        # outcome learner choices.
        x_learners["x_" + name] = XLearnerWrapper(X, t, y, cv, 
                                     outcome_learner = base_learner, 
                                     effect_learner = base_learners[effect_learner], 
                                     params_treat = t_learners["t_" + name] \
                                            .treatment_outcome_learner.get_params(),
                                     params_control = t_learners["t_" + name] \
                                            .control_outcome_learner.get_params(),
                                     effect_param_grid = param_grids[effect_learner])
        if tune:
            print("Tuning " + "x_" + name)
            x_learners["x_" + name].tune_params(n_iter)
    
    # Add R-learner estimators. First make metalearners out of all the
    # possible combinations of base learners, and then select those we want
    # to keep
    r_learners_all = {}
    for name_1 in ["lasso", "rf", "xgb"]:
        for name_2 in ["lasso", "rf", "xgb"]:
            r_learners_all["r_" + name_1 + name_2] = RLearnerWrapper(X, t, y, cv,
                                        outcome_learner = base_learners[name_1], 
                                        effect_learner = base_learners[name_2],
                                        outcome_param_grid = param_grids[name_1],
                                        effect_param_grid = param_grids[name_2])
    r_learners = {}
    r_learner_names = ["r_lassolasso", "r_rfrf", "r_lassorf", "r_lassoxgb"]
    for name in r_learner_names:
        r_learners[name] = r_learners_all[name]
        if tune:
            print("Tuning " + name)
            r_learners[name].tune_params(n_iter)
    library = s_learners
    library.update(t_learners)
    library.update(x_learners)
    library.update(r_learners)       
    
    # If we don't tune the estimators (i.e. if no param_grids are supplied), then
    # set them using the supplied tune_params dictionary. Also add the tree-based
    # estimators (that we can't tune) into the library
    if not tune:
        for estimator_name, estimator in library.items():
            estimator.set_params(tuned_params[estimator_name])
        library["causal_tree_1"] = CausalTreeWrapper(X, t, y, cv,
                                       params = {"min_samples_leaf" : 50})
        library["causal_tree_2"] = CausalTreeWrapper(X, t, y, cv, 
                                       params = {"min_samples_leaf" : 200})
        library["causal_forest_1"] = CausalForestWrapper(X, t, y, cv, 
                                         params = {"min_size" : 50, 
                                                   "feature_split" : 0.33, 
                                                   "max_depth" : 6,
                                                   "bootstrap" : True})
        library["causal_forest_2"] = CausalForestWrapper(X, t, y, cv,
                                         params = {"min_size" : 200, 
                                                   "feature_split" : 0.33, 
                                                   "max_depth" : 6,
                                                   "bootstrap" : True})
    
    return library


def fit_estimator_libraries(DATA_PATH, original_features, outcome_name, 
                            base_learners, tuned_params, n_splits, 
                            perturbed = True, num_cv_splits = 2, 
                            alternate_outcome = None):
    """
    Create fitted estimator library/libraries. If perturbed flag is set to True,
    returns a collection of libraries, one fitted for each perturbed data set,
    otherwise returns the library fit on the original dataset only.
    
    Parameters
    ----------
    DATA_PATH: string
        The path to where the processed data is stored.
    original_features: list of strings
        The features that were used to tune the hyperparameters of the 
        estimators.
    outcome_name: string
        The label of the outcome we are concerned with
    base_learners: dict of sklearn estimators
        Dictionary of sklearn estimators to use as base learners in the 
        meta-learners.
    n_splits: int
        Number of folds for CV
    perturbed: bool
        A flag for whether to fit estimators also on perturbed datasets
    num_cv_splits: int
        Number of additional CV splits to do as data perturbations.
    alternate_outcome: string
        The label of an alternate outcome which is a 'data-driven'
        perturbation of the outcome of interest.
    
    Returns
    -------
    fitted_libraries: mapping of mappings of string to CATEEstimatorWrapper
       A dictionary of libraries of fitted CATE estimators
    """    
    
    FILE_PATH = os.path.join(DATA_PATH, outcome_name, "trainval_data.csv")
    fitted_libraries = {} # Dictionary of dictionaries of methods
        
    # Fit estimators on unperturbed data
    trainval_df = prepare_df(FILE_PATH, original_features, outcome_name)
    X, t, y = separate_vars(trainval_df, outcome_name)
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 405)
    fitted_libraries["pert_none"] = make_estimator_library(X, t, y, cv, 
                             base_learners, tuned_params = tuned_params)
    for estimator in tqdm(fitted_libraries["pert_none"].values()):
        np.random.seed(123123)
        estimator.fit()

    if perturbed:
        # 1. Fit methods with different CV splits
        for i in range(num_cv_splits):
            cv_new = StratifiedKFold(n_splits = n_splits, 
                                   shuffle = True, random_state = 7*i)
            fitted_libraries["pert_cv_" + str(i)] = \
                make_estimator_library(X, t, y, cv_new, base_learners,
                                       tuned_params = tuned_params)
            for estimator in tqdm(fitted_libraries["pert_cv_" + str(i)].values()):
                np.random.seed(123123)
                estimator.fit()
                
        # 2. Use time-based CV split
        trainval_df = prepare_df(FILE_PATH, original_features + ["RAND_DT"], 
                                 outcome_name)
        trainval_df.sort_values("RAND_DT", inplace = True)
        trainval_df.drop(columns = ["RAND_DT"], inplace = True)
        X, t, y = separate_vars(trainval_df, outcome_name)
        cv_time = KFold(n_splits = n_splits, shuffle = False)
        fitted_libraries["pert_cv_time"] = \
            make_estimator_library(X, t, y, cv_time, base_learners,
                                   tuned_params = tuned_params)
        for estimator in tqdm(fitted_libraries["pert_cv_time"].values()):
            np.random.seed(123123)
            estimator.fit()
        
        # 3. Use overweight instead of obese
        new_features = copy.deepcopy(original_features)
        new_features.remove("obese")
        new_features.append("overweight")
        trainval_df = prepare_df(FILE_PATH, new_features, outcome_name)
        X, t, y = separate_vars(trainval_df, outcome_name)
        fitted_libraries["pert_overweight"] = \
            make_estimator_library(X, t, y, cv, base_learners,
                                   tuned_params = tuned_params)
        for estimator in tqdm(fitted_libraries["pert_overweight"].values()):
            np.random.seed(123123)
            estimator.fit()
        
        # 4. Use different elderly definition
        new_features = copy.deepcopy(original_features)
        new_features.remove("elderly_65_adj")
        new_features.append("elderly_60_adj")
        trainval_df = prepare_df(FILE_PATH, new_features, outcome_name)
        X, t, y = separate_vars(trainval_df, outcome_name)
        fitted_libraries["pert_elderly_60"] = \
            make_estimator_library(X, t, y, cv, base_learners,
                                   tuned_params = tuned_params)
        for estimator in tqdm(fitted_libraries["pert_elderly_60"].values()):
            np.random.seed(123123)
            estimator.fit()
        
        # 5. Use different outcome
        if alternate_outcome is not None:
            trainval_df = prepare_df(FILE_PATH, original_features, 
                                     alternate_outcome)
            X, t, y = separate_vars(trainval_df, alternate_outcome)
            fitted_libraries["pert_outcome"] = \
                make_estimator_library(X, t, y, cv, base_learners,
                                       tuned_params = tuned_params)
            for estimator in tqdm(fitted_libraries["pert_outcome"].values()):
                np.random.seed(123123)
                estimator.fit()
            
    return fitted_libraries

#============================================================#
##### 2. Functions for validating the estimators         #####
#============================================================#

def get_bin_CATEs(CATE_estimator, fold, n_bins = 5, kind = "val", 
                  return_std = False):
    """
    Using predicted ITE quantiles on the training fold as thresholds, put the 
    individuals into n_bins of roughly equal size. Compute the Neyman estimate 
    for the CATEs over each bin.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        CATE estimator to be used to predict ITEs and hence define the bins
    fold: int in {0,1,2,3}
        The holdout fold number to use
    n_bins: int
        The number of bins to use
    kind: string in {"val", "train", "all"}
        Flag to indicate whether to restrict to samples in the validation set, 
        training set, or the entire training-validation set when binning 
        individuals.
    return_std: bool
        Flag for whether or not to also return the standard deviation estimate.
        
    Returns
    -------
    bin_CATEs_: array of shape (n_bins,)
        The Neyman estimate for the CATEs over each bin
    bin_CATEs_std_ (optional): array of shape (n_bins,)
        The std estimate for the Neyman CATE estimator over each bin.
    """
    assert fold in {0, 1, 2, 3}
    assert kind in {"val", "train", "all"}
    
    q_values = np.linspace(0, 1, n_bins+1)
    bin_CATEs_ = []
    bin_CATEs_std_ = []
    y = CATE_estimator.y
    t = CATE_estimator.t
    for i in range(n_bins):
        subgroup_indicator = CATE_estimator.results[fold] \
                    .get_subgroup_indicator(q_values[i], q_values[i+1], kind)
        CATE_ = get_subgroup_CATE(y, t, subgroup_indicator)
        bin_CATEs_.append(CATE_)
        if return_std:
            CATE_std_ = get_subgroup_CATE_std(y, t, subgroup_indicator)
            bin_CATEs_std_.append(CATE_std_)
        else:
            pass
        
    bin_CATEs_ = np.array(bin_CATEs_)
    bin_CATEs_std_= np.array(bin_CATEs_std_)
    
    if return_std:
        return (bin_CATEs_, bin_CATEs_std_)
    else:
        return bin_CATEs_

def get_bin_model_CATEs(CATE_estimator, fold, n_bins = 5, kind = "val",
                        return_std = False):
    """
    Using predicted ITE quantiles on the training fold as thresholds, put the 
    individuals into n_bins of roughly equal size. Compute the model estimate 
    for the CATEs over each bin (i.e. average the predicted ITEs over each bin.)
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs and hence define the 
        bins
    fold: int in {0,1,2,3}
        The holdout fold number to use
    n_bins: int
        The number of bins to use
    kind: string in {"val", "train", "all"}
        Flag to indicate whether to restrict to samples in the validation set, 
        training set, or the entire training-validation set when binning 
        individuals.
    return_std: bool
        Flag for whether or not to also return the standard deviation estimate.
        
    Returns
    -------
    bin_CATEs_: array of shape (n_bins,)
        The model estimate for the CATEs over each bin  
    bin_CATEs_std_ (optional): array of shape (n_bins,)
        The std estimate for the Neyman CATE estimator over each bin.
    """
    assert fold in {0, 1, 2, 3}
    assert kind in {"val", "train", "all"}
    
    q_values = np.linspace(0, 1, n_bins+1)
    bin_CATEs_ = []
    bin_CATEs_std_ = []
    for i in range(n_bins):
        subgroup_indicator = CATE_estimator.results[fold] \
                    .get_subgroup_indicator(q_values[i], q_values[i+1], kind)
        tau_values = CATE_estimator.results[fold].tau[subgroup_indicator]
        CATE_ = tau_values.mean()
        bin_CATEs_.append(CATE_)
        if return_std:
            CATE_std_ = tau_values.std(ddof = 1) / np.sqrt(len(tau_values))
            bin_CATEs_std_.append(CATE_std_)
        else:
            pass
    bin_CATEs_ = np.array(bin_CATEs_)

    if return_std:
        return (bin_CATEs_, bin_CATEs_std_)
    else:
        return bin_CATEs_
    
def get_calibration_score(model_bin_CATEs_, Neyman_bin_CATEs_, 
                          n_bins_selected = None, dir_neg = True):
    """
    Compute the l1 calibration score for a CATE model. Calibration score is
    defined in the paper.
    
    Parameters
    ----------
    model_bin_CATEs_: array of shape (n_bins,)
        The model estimate for the CATEs over each bin
    Neyman_bin_CATEs_: array of shape (n_bins,)
        The Neyman estimate for the CATEs over each bin
    n_bins_selecteed: int or None
        The number of bins to select in computing the score. If None,
        then all bins are used
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
        
    Returns
    -------
    calibration_score: float
        The l1 calibration score for the CATE model
    """
    
    if n_bins_selected is None:
        n_bins_selected = len(Neyman_bin_CATEs_)

    if dir_neg:
        calibration_score = \
            np.abs((Neyman_bin_CATEs_-model_bin_CATEs_)[:n_bins_selected]).sum()
    else:
        calibration_score = \
            np.abs((Neyman_bin_CATEs_-model_bin_CATEs_)[-n_bins_selected:]).sum()
    return calibration_score
    
def get_cr2_score(model_bin_CATEs_, Neyman_bin_CATEs_, ATE_,
                  n_bins_selected = None, dir_neg = True):
    """
    Compute the cr2 score for a CATE model. The cr2 score is
    defined in the paper.
    
    Parameters
    ----------
    model_bin_CATEs_: array of shape (n_bins,)
        The model estimate for the CATEs over each bin
    Neyman_bin_CATEs_: array of shape (n_bins,)
        The Neyman estimate for the CATEs over each bin
    ATE_: float
        The Neyman ATE estimate
    n_bins_selecteed: int or None
        The number of bins to select in computing the score. If None,
        then all bins are used
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
        
    Returns
    -------
    cr2_score: float
        The cr2 score for the CATE model
    """
    
    cal_score_model = get_calibration_score(model_bin_CATEs_, Neyman_bin_CATEs_,
                                            n_bins_selected, dir_neg)
    cal_score_ATE_ = get_calibration_score(ATE_, Neyman_bin_CATEs_,
                                           n_bins_selected, dir_neg)
    
    return 1 - cal_score_model/cal_score_ATE_

def get_r_values(fitted_libraries, compare_across = "folds",
                 estimator = None, fold = None, dropped_estimators = []):
    """
    Compute the Pearson correlation values for the predicted ITEs from a 
    collection of models. There are 3 options which can be selected using 
    the compare_across flag: If compare_across == "folds", we use the
    collection of models we get by fitting one estimator using different 
    training folds. If compare_across == "estimators", we use the collection of
    models we get by fitting different estimators using the same training fold.
    Finally, if compare_across == "all", we use the collection of all models
    obtained by varying both the fold and the estimator.
    
    Parameters
    ----------
    fitted_libraries: mapping of mappings of string to CATEEstimatorWrapper
        A dictionary of libraries of fitted CATE estimators
    compare_across: string in {"folds", "estimators", "all"}
        Flag to decide output of function
    estimator: string
        Name of the estimator to be used (if compare_across == "folds")
    fold: int
        Fold number to use (if compare_across == "estimators")
    dropped_estimators: list of strings
        The names of the estimators not to include when generating the results.
    
    Returns
    -------
    correlations: array-like
        An array containing the Pearson correlation values, with one entry for
        each pair of distinct models.
    """
    
    assert compare_across in ["folds", "estimators", "all"]
    libraries = [fitted_libraries["pert_none"], fitted_libraries["pert_cv_0"],
                 fitted_libraries["pert_cv_1"]]
    n_splits = fitted_libraries["pert_none"]["t_lasso"].n_splits
    model_CATEs_df = pd.DataFrame({})
    if compare_across == "folds":
        for i, library in enumerate(libraries):
            for fold in range(n_splits):
                model_CATEs_df[i*n_splits + fold] = library[estimator]. \
                                                            results[fold].tau
    elif compare_across == "estimators":
        fold = fold % 4
        library = libraries[fold // 4]
        for name, estimator in library.items():
            if name not in dropped_estimators:
                model_CATEs_df[name] = estimator.results[fold].tau
    else:
        for name in libraries[0].keys():
            if name not in dropped_estimators:
                for i, library in enumerate(libraries):
                    for fold in range(n_splits):
                        model_CATEs_df[name + f"_fold_{i*n_splits + fold}"] \
                                            = library[name].results[fold].tau
    correlations_df = model_CATEs_df.corr()
    m = correlations_df.shape[0]
    correlations = correlations_df.values[np.triu_indices(m, 1)]
    
    return correlations

def get_overlap_values(fitted_libraries, q_bot, q_top, compare_across = "folds",
                       estimator = None, fold = None, dropped_estimators = []):
    """
    Compute the percentage overlap of subgroups defined using quantiles of ITEs 
    from a collection of models. There are 3 options which can be selected using 
    the compare_across flag: If compare_across == "folds", we use the
    collection of models we get by fitting one estimator using different 
    training folds. If compare_across == "estimators", we use the collection of
    models we get by fitting different estimators using the same training fold.
    Finally, if compare_across == "all", we use the collection of all models
    obtained by varying both the fold and the estimator.
    
    Parameters
    ----------
    fitted_libraries: mapping of mappings of string to CATEEstimatorWrapper
        A dictionary of libraries of fitted CATE estimators
    q_bot: float
        q value for bottom quantile cutoff
    q_top: float
        q value for top quantile cutoff
    compare_across: string in {"folds", "estimators", "all"}
        Flag to decide output of function
    estimator: string
        Name of the estimator to be used (if compare_across == "folds")
    fold: int
        Fold number to use (if compare_across == "estimators")
    dropped_estimators: list of strings
        The names of the estimators not to include when generating the results.
    
    Returns
    -------
    overlaps: array-like
        An array containing the percentage overlap values, with one entry for
        each pair of distinct models.
    """
    
    assert compare_across in ["folds", "estimators", "all"]
    libraries = [fitted_libraries["pert_none"], fitted_libraries["pert_cv_0"],
                 fitted_libraries["pert_cv_1"]]
    n_splits = fitted_libraries["pert_none"]["t_lasso"].n_splits
    subgroup_indicators_df = pd.DataFrame({})
    if compare_across == "folds":
        for i, library in enumerate(libraries):
            for fold in range(n_splits):
                subgroup_indicators_df[i*n_splits + fold] = library[estimator] \
                           .results[fold].get_subgroup_indicator(q_bot, q_top, "all")
    elif compare_across == "estimators":
        fold = fold % 4
        library = libraries[fold // 4]
        for name, estimator in library.items():
            if name not in dropped_estimators:
                subgroup_indicators_df[name] = estimator.results[fold] \
                                .get_subgroup_indicator(q_bot, q_top, "all")
    else:
        for name in libraries[0].keys():
            if name not in dropped_estimators:
                for i, library in enumerate(libraries):
                    for fold in range(n_splits):
                        subgroup_indicators_df[name + f"_fold_{i*n_splits + fold}"] \
                                            = library[name].results[fold] \
                                .get_subgroup_indicator(q_bot, q_top, "all")
    n_samples = len(subgroup_indicators_df.index)
    subgroup_size = n_samples * (q_top-q_bot)
    subgroup_indicators_df = subgroup_indicators_df.astype(int)
    overlaps_df = (subgroup_indicators_df.T @ subgroup_indicators_df) /  subgroup_size 
    m = overlaps_df.shape[0]
    overlaps = overlaps_df.values[np.triu_indices(m, 1)]
    
    return overlaps

def check_if_sequence_increasing(sequence):
    """
    Check whether pairs of successive entries of a sequence are in
    increasing order, and return the resulting list of boolean values.
    
    Parameters
    ----------
    sequence: array-like of shape (n_elements,)
        A sequence whose entries are to be checked.
    
    Returns
    -------
    is_increasing: array-like of shape (n_elements-1,)
        A list of boolean values each indicating whether the corresponding
        pair of entries in sequence is increasing.
    """
    differences = []
    for i in range(len(sequence)-1):
        differences.append(sequence[i+1] - sequence[i])
    is_increasing = np.array(differences) > 0
    
    return is_increasing

def get_estimator_monotonicity_results(CATE_estimator, n_bins = 5, 
                                       dir_neg = True):
    """
    After putting individuals into bins using predicted ITE quantiles, 
    compute the Neyman CATE estimates for each bin using validation data. 
    Check whether the pairs of successive bins have CATE estimates in the
    right order, returning the results in a data frame with each column
    corresponding to a pairwise comparison, and each row corresponding to
    a choice of training fold. Also add a column for whether the extremal
    bin of interest has the extremal validation Neyman CATE estimate.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs and hence define the 
        bins
    n_bins: int
        The number of bins to use
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
        
    Returns
    -------
    monotonicity_df: data frame
        Data frame containing monotonicity results for the estimator.
    """
    def is_bin1_smallest(bin_CATEs):
        return bin_CATEs.argmin() == 0
    
    def is_lastbin_largest(bin_CATEs):
        return bin_CATEs.argmax() == n_bins-1
    
    if dir_neg:
        func = is_bin1_smallest
        col_name = "first_bin_is_min"
    else:
        func = is_lastbin_largest
        col_name = "last_bin_is_max"
    
    q_values = np.linspace(0, 1, n_bins+1)
    col_names = [f"[{q_values[i]:.1f},{q_values[i+1]:.1f}] vs \
                 [{q_values[i+1]:.1f}, {q_values[i+2]:.1f}]" 
                 for i in range(n_bins-1)] + [col_name]
    monotonicity_df = pd.DataFrame({}, columns = col_names)
    for fold in range(CATE_estimator.n_splits):
        bin_CATEs = get_bin_CATEs(CATE_estimator, fold, n_bins, "val")
        is_increasing = list(check_if_sequence_increasing(bin_CATEs))
        is_increasing.append(func(bin_CATEs))
        monotonicity_df = monotonicity_df.append(pd.Series(is_increasing, 
                                                           index = col_names),
                                                 ignore_index = True) \
                                                    .astype(int)
    return monotonicity_df

def get_estimator_monotonicity_results_v2(CATE_estimator, q_values, 
                                          dir_neg = True):
    """
    For each q value, we define the subgroup by thresholding using the
    corresponding quantile for the predicted ITE values. Compute the Neyman 
    CATE estimates for both the subgroup and its complement using validation 
    data and check whether they are in the right order. We return the results 
    in a data frame with each column corresponding to a q value, and each row 
    corresponding to a choice of training fold.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs and hence define the 
        bins
    q_values: list
        The q values whose quantiles we will use as thresholds to define
        the reference QBTS.
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
        
    Returns
    -------
    monotonicity_df: data frame
        Data frame containing monotonicity results for the estimator.
    """
    
    if dir_neg:
        pass
    else:
        q_values = [1-q for q in q_values]
    col_names = [f"[{0.},{q:.1f}] vs [{q}, 1.]" for q in q_values] 
    
    monotonicity_df = pd.DataFrame({}, columns = col_names)
    y = CATE_estimator.y
    t = CATE_estimator.t
    for fold in range(CATE_estimator.n_splits):
        is_increasing = []
        for q in q_values:
            subgroup_indicator = CATE_estimator.results[fold] \
                            .get_subgroup_indicator(0, q, "val")
            bot_CATE = get_subgroup_CATE(y, t, subgroup_indicator)
            top_CATE = get_subgroup_CATE(y, t, ~subgroup_indicator)
            is_increasing.append(bot_CATE < top_CATE)
        monotonicity_df = monotonicity_df.append(pd.Series(is_increasing, 
                                                           index = col_names),
                                                 ignore_index = True) \
                                                        .astype(int)
    return monotonicity_df
        

def get_monotonicity_results(fitted_libraries, top_estimator_names = None, 
                             n_bins = 5, q_values = None, dir_neg = True, 
                             dropped_estimators = []):
    """
    Aggregate the monotonicity results over all estimators. Returns a data
    frame whose columns are the same as those returned by 
    get_estimator_monotonicity_results, but each row now corresponds to an
    estimator, with entries the means taken over all (12) folds for that estimator.
    If q_values is not None, we compare the bottom quantile-based subgroup
    against its complement, otherwise we compare n_bins pairwise.
    
    Parameters
    ----------
    fitted_libraries: mapping of mappings of string to CATEEstimatorWrapper
        A dictionary of libraries of fitted CATE estimators
    top_estimator_names: list of strings
        The names of the estimators that are of special interest.
    q_values: list
        The q values whose quantiles we will use as thresholds to define
        the reference QBTS.
    n_bins: int
        The number of bins to use
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
    dropped_estimators: list of strings
        The names of the estimators not to include when generating the results.
        
    Returns
    -------
    aggregated_monotonicity_df: data frame
         Data frame containing the aggregated monotonicity results.
    """

    aggregated_monotonicity_df = pd.DataFrame({})
    libraries = [fitted_libraries["pert_none"], fitted_libraries["pert_cv_0"],
                 fitted_libraries["pert_cv_1"]]
    for name in libraries[0].keys():
        if name not in dropped_estimators:
            estimator_monotonicity_dfs = []
            for library in libraries:
                if q_values is not None:
                    monotonicity_results = get_estimator_monotonicity_results_v2( \
                                            library[name], q_values, dir_neg)
                else:
                    monotonicity_results = get_estimator_monotonicity_results( \
                                            library[name], n_bins, dir_neg)
                estimator_monotonicity_dfs.append(monotonicity_results)
            estimator_monotonicity_df = pd.concat(estimator_monotonicity_dfs)
            estimator_monotonicity_entry = estimator_monotonicity_df \
                                                        .mean(axis = 0)
            aggregated_monotonicity_df[name] = estimator_monotonicity_entry
    aggregated_monotonicity_df.index = estimator_monotonicity_entry.index
    aggregated_monotonicity_df["mean"] = aggregated_monotonicity_df.mean(axis = 1)
    if top_estimator_names is not None:
        aggregated_monotonicity_df[f"top{len(top_estimator_names)}_mean"] = \
            aggregated_monotonicity_df.filter(items = top_estimator_names) \
            .mean(axis = 1)
    aggregated_monotonicity_df = aggregated_monotonicity_df.T
    
    return aggregated_monotonicity_df


def get_top_subgroup_stability(CATE_estimator, q0, q_values, dir_neg = True):
    """
    Computes the average percentage inclusion of a quantile-based top subgroup 
    (QBTS) with respect to q0 is included in another QBTS with respect to q, 
    when both arise from CATE estimators trained on different training folds. We 
    compute this for every q in q_values and output the results in a data frame.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs and hence define the 
        QBTS's
    q0: float
        The q value used as a threshold to define the QBTS whose percentage
        inclusion we want to compute.
    q_values: list
        The q values whose quantiles we will use as thresholds to define
        the reference QBTS.
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
        
    Returns
    -------
    stability_df: data frame
        A data frame
    
    """
    n_samples = len(CATE_estimator.y)
    n_folds = CATE_estimator.n_splits
    
    def get_subgroup_indicators(q):
        subgroup_indicators = np.zeros((n_samples, n_folds))
        t_statistics = np.zeros(n_folds)
        if dir_neg:
            q_bot = 0
            q_top = q
        else:
            q_bot = 1-q
            q_top = 1
        sg_size = n_samples * q
        for fold, result in CATE_estimator.results.items():
            subgroup_indicator = result \
                        .get_subgroup_indicator(q_bot, q_top, "all")
            subgroup_indicators[:,fold] = subgroup_indicator
        subgroup_indicators = subgroup_indicators.astype(int)
        return subgroup_indicators
    
    def get_frac_covered(indicator1, indicator2):
        frac_covered = (indicator1 & indicator2).sum() / indicator1.sum()
        return frac_covered
    
    frac_covered_means = []
    frac_covered_stds = []
    q0_indicators = get_subgroup_indicators(q0)
    for q in q_values:
        q_indicators = get_subgroup_indicators(q)
        frac_covered_mat = q0_indicators.T @ q_indicators / q0_indicators.sum(axis = 0)
        frac_covered = frac_covered_mat[np.triu_indices(n_folds, 1)]
        frac_covered_means.append(frac_covered.mean())
        frac_covered_stds.append(frac_covered.std())
    stability_df = pd.DataFrame({"mean coverage" : frac_covered_means,
                                 "coverage std" : frac_covered_stds},
                                index = q_values)
    
    return stability_df
    


#============================================================#
##### 2a. Functions for making plots for validation      #####
#============================================================#
    
def get_calibration_plot_data(CATE_estimator, fold, n_bins = 5, kind = "val"):
    """
    Produce a data frame with the data needed to make a calibration plot.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs and hence define the 
        bins
    fold: int in {0,1,2,3}
        The holdout fold number to use
    n_bins: int
        The number of bins to use
    kind: string in {"val", "train", "all"}
        Flag to indicate whether to restrict to samples in the validation set, 
        training set, or the entire training-validation set when binning 
        individuals.
        
    Returns
    -------
    cal_plot_data: data frame
        Data frame with the data needed to make a calibration plot.
    """
    Neyman_CATEs, Neyman_CATEs_std = get_bin_CATEs(CATE_estimator, fold, 
                                                   n_bins, kind, True)
    model_CATEs, model_CATEs_std = get_bin_model_CATEs(CATE_estimator, fold, 
                                                       n_bins, kind, True)
    q_values = np.linspace(0, 1, n_bins+1)
    bin_labels = [f"[{q_values[i]:.1f},{q_values[i+1]:.1f}]" for i in range(n_bins)]
    cal_plot_data = pd.DataFrame({"Neyman_CATEs" : Neyman_CATEs,
                                  "Neyman_CATEs_std" : Neyman_CATEs_std,
                                  "model_CATEs" : model_CATEs,
                                  "model_CATEs_std" : model_CATEs_std},
                                 index = bin_labels)
    
    return cal_plot_data

def get_cr2_plot_data(CATE_estimator, n_bins = 5,
                      n_bins_selected = None, dir_neg = True):
    """
    Produce a data frame with the data needed to make a cr2 plot.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs and hence define the 
        bins
    n_bins: int
        The number of bins to use
    Returns
    -------
    cr2_plot_data: data frame
        Data frame with the data needed to make a cr2 plot.
    """
    
    n_splits = CATE_estimator.n_splits
    cr2_scores_train = []
    cr2_scores_val = []
    for fold in range(n_splits):
        Neyman_CATEs_train = get_bin_CATEs(CATE_estimator, fold, 
                                           n_bins, "train")
        model_CATEs_train = get_bin_model_CATEs(CATE_estimator, fold, 
                                                n_bins, "train")
        ATE_train = get_Neyman_ATE(CATE_estimator.y[CATE_estimator.results[fold] \
                                              .train_indices], 
                             CATE_estimator.t[CATE_estimator.results[fold] \
                                              .train_indices])
        cr2_score = get_cr2_score(model_CATEs_train, Neyman_CATEs_train, ATE_train,
                                  n_bins_selected, dir_neg)
        cr2_scores_train.append(cr2_score)
        Neyman_CATEs_val = get_bin_CATEs(CATE_estimator, fold, n_bins, "val")
        model_CATEs_val = get_bin_model_CATEs(CATE_estimator, fold, n_bins, "val")
        ATE_val = get_Neyman_ATE(CATE_estimator.y[CATE_estimator.results[fold] \
                                                  .val_indices], 
                             CATE_estimator.t[CATE_estimator.results[fold] \
                                              .val_indices])
        cr2_score = get_cr2_score(model_CATEs_val, Neyman_CATEs_val, ATE_val,
                                  n_bins_selected, dir_neg)
        cr2_scores_val.append(cr2_score)

    cr2_plot_data = pd.DataFrame({"cr2_train" : cr2_scores_train,
                                  "cr2_val" : cr2_scores_val})
    
    return cr2_plot_data

#============================================================#
##### 3. Functions for ranking the estimators            #####
#============================================================#

def get_estimator_scores(fitted_libraries, q_values, 
                         dir_neg = True, kind = "val"):
    """
    Compute t-statistic values for top subgroups. That is, subgroups 
    defined by selecting all individuals whose predicted ITE falls below 
    or above the quantiles corresponding to the q values in the q_values 
    argument.
    
    Parameters
    ----------
    fitted_libraries: mapping of mappings of string to CATEEstimatorWrapper
        A dictionary of libraries of fitted CATE estimators
    q_values: list
        The q values whose quantiles we will use as thresholds to define
        the top subgroups.
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
    kind: string
        flag in {"val", "train", "all"} to indicate whether to restrict
        to samples in the validation set, training set, or the entire
        training-validation set.
        
    Returns
    -------
    scores_df: data frame of shape (n_estimators, n_quantiles * 
                                            n_perturbations * n_folds)
        A data frame containing the t-statistics of the top subgroups
        
    """
    assert kind in ["val", "train", "all"]
    
    n_splits = fitted_libraries["pert_none"]["t_lasso"].n_splits
    estimator_names = list(fitted_libraries["pert_none"].keys())
    scores_df = pd.DataFrame({"estimator" : estimator_names})

    for perturbation_name, library in fitted_libraries.items():
        y = library["t_lasso"].y
        t = library["t_lasso"].t
        for fold_no in range(n_splits):
            # Compute dataset indicator
            if kind == "val":
                dataset_indicator = library["t_lasso"].results[fold_no] \
                                                            .val_indicator
            elif kind == "train":
                dataset_indicator = library["t_lasso"].results[fold_no] \
                                                            .train_indicator
            else:
                dataset_indicator = None
                
            for q in q_values:
                scores = []
                for estimator in library.values():
                    # Compute subgroup indicator
                    if dir_neg:
                        subgroup_indicator = estimator.results[fold_no] \
                                    .get_subgroup_indicator(0, q, kind)
                    else:
                        subgroup_indicator = estimator.results[fold_no] \
                                    .get_subgroup_indicator(1-q, 1, kind)
                    # Compute t-statistic
                    t_statistic = get_subgroup_t_statistic(y, t, 
                                        subgroup_indicator, dataset_indicator)
                    scores.append(t_statistic)
                scores_df["pert=" + perturbation_name[5:] 
                          + f"/q={q}/fold={fold_no}"] = scores
                    
    return scores_df.set_index("estimator")

def aggregate_estimator_scores(scores_df, group_by = "perturbation",
                               aggregation_func_name = "mean", 
                               k = 10, dir_neg = True):
    """
    Provides a convenient interface to aggregate the the t-statistic scores 
    for each estimator over a choice of dimenssion and using a choice of
    aggregation function.
    
    Parameters
    ----------
    scores_df: data frame of shape (n_estimators, n_quantiles * 
                                            n_perturbations * n_folds)
        A data frame containing the t-statistics of the top subgroups. We treat
        these t-statistics as scores.
    group_by: string in {"perturbation", "q_value", "none"}
        A flag for the dimension to perform aggregation over
    aggregation_func_name: string in {"mean", "rank_mean", "top_k_freq"}
        A flag for the aggregation function to use
        mean: Computes the mean of the scores
        rank_mean: Computes the rank of the estimator over a column of scores_df
            and then computes the mean of these ranks
        top_k_freq: For a given choice of k, computes the rank of the estimator 
            over a column of scores_df and then computes the frequency with which
            it is ranked among the top k estimators.
    k: int
        Parameter for top_k_freq
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect. Hence, whether lower scores are preferred, 
        or higher scores are preferred.
        
    Returns
    -------
    aggregated_df: data frame
        A data frame with the aggregated scores for the estimators.
    """
    
    assert group_by in {"perturbation", "q_value", "none"}
    assert aggregation_func_name in {"mean", "std", "rank_mean", "top_k_freq"}
    
    estimator_names = scores_df.index
    q_values = list({float(item.split("/")[1][2:]) for item in scores_df.columns})
    q_values = sorted(q_values)
    n_splits = max({int(item.split("/")[2][5:]) for item in scores_df.columns}) + 1
    
    # Define aggregation functions
    def get_mean(scores):
        return scores.apply(lambda row : row.mean(), axis = 1)
    
    def get_std(scores):
        return scores.apply(lambda row : row.std(), axis = 1)
    
    def get_rank_mean(scores):
        return scores.rank(ascending = dir_neg) \
                .apply(lambda row : row.mean(), axis = 1)
    
    def make_top_k_freq(k):
        def get_top_k_freq(scores):
            return (scores.rank(ascending = dir_neg) <= k) \
                    .apply(lambda row : row.mean(), axis = 1)
        return get_top_k_freq
    
    # Dispatch dictionary
    aggregation_funcs = {"mean" : get_mean, 
                         "std" : get_std,
                         "rank_mean" : get_rank_mean,
                         "top_" + str(k) + "_freq" : make_top_k_freq(k)
                        }
    if aggregation_func_name == "top_k_freq":
        aggregation_func_name = "top_" + str(k) + "_freq"
    aggregation_func = aggregation_funcs[aggregation_func_name]
    
    # Perform aggregations depending on group_by argument
    aggregated_df = pd.DataFrame({}, index = estimator_names)
    if group_by == "perturbation":
        perturbation_names = set([ item.split("/")[0] for item 
                                  in scores_df.columns])
        step = n_splits * len(q_values)
        for perturbation_name in perturbation_names:
            perturbation_scores = scores_df.filter(regex=perturbation_name)
            aggregated_df[perturbation_name] = \
                        aggregation_func(perturbation_scores)
    elif group_by == "q_value":
        for q in q_values:
            q_scores = scores_df.filter(regex=str(q))
            aggregated_df[q] = aggregation_func(q_scores)
    elif group_by == "none":
        for func_name, func in aggregation_funcs.items():
            aggregated_df[func_name] = func(scores_df)
    else:
        pass
    
    return aggregated_df

def get_t_r2_statistics(scores_df):
    """
    Get R2 scores for t-statistics w.r.t. stratification by q-value, perturbation
    type, or by estimator.
    
    Parameters
    ----------
    scores_df: data frame of shape (n_estimators, n_quantiles * 
                                            n_perturbations * n_folds)
        A data frame containing the t-statistics of the top subgroups. We treat
        these t-statistics as scores.
        
    Returns
    -------
    r2_results_df: data frame
        A data frame with the R2 scores.    
    """
    
    SS_tot = scores_df.values.var() * scores_df.shape[0] * scores_df.shape[1]
    
    # Get R2 for q values
    q_values = list({float(item.split("/")[1][2:]) for item in scores_df.columns})
    SS_res_q = 0
    for q in q_values:
        filtered_df = scores_df.filter(regex = str(q))
        SS_res_q += filtered_df.values.var() * filtered_df.shape[0] \
                                             * filtered_df.shape[1]
    r2_q = 1 - SS_res_q / SS_tot
    
    # Get R2 for perturbations
    perturbation_names = set([ item.split("/")[0] for item in scores_df.columns])
    SS_res_pert = 0
    for name in perturbation_names:
        filtered_df = scores_df.filter(regex = name)
        SS_res_pert += filtered_df.values.var() * filtered_df.shape[0] \
                                                * filtered_df.shape[1]
    r2_pert = 1 - SS_res_pert / SS_tot
    
    # Get R2 for estimators
    perturbation_names = set([ item.split("/")[0] for item in scores_df.columns])
    SS_res_est = 0
    for estimator in scores_df.index:
        SS_res_est += scores_df.loc[estimator].values.var() * scores_df.shape[1]
    r2_est = 1 - SS_res_est / SS_tot
    
    r2_results_df = pd.DataFrame({"R2" : [r2_q, r2_pert, r2_est]}, 
                                 index = ["q values", "perturbations", 
                                          "estimators"])
    
    return r2_results_df

def make_top_ensemble(fitted_libraries, top_estimator_names):
    """
    Create an ensemble CATE estimator using a given list of fitted CATE estimators.
    
    Parameters
    ----------
    fitted_libraries: mapping of mappings of string to CATEEstimatorWrapper
        A dictionary of libraries of fitted CATE estimators
    top_estimator_names: list of strings
        The names of the estimators to be used to create the ensemble.
        
    Returns
    -------
    top_ensemble: EnsembleCATEEstimatorWrapper object
        Wrapper class objects for the fitted ensembles with results from all 3 CV
        splits (4 folds each).
    
    """
    library_names = ["pert_none", "pert_cv_0", "pert_cv_1"]
    ensemble_list = []
    for library_name in library_names:
        library = fitted_libraries[library_name]
        top_estimators = [library[estimator_name] for estimator_name in 
                          top_estimator_names]
        ensemble_version = EnsembleCATEEstimatorWrapper(top_estimators)
        ensemble_version.fit()
        ensemble_list.append(ensemble_version)
    top_ensemble = combine_estimator_versions(ensemble_list)
        
    return top_ensemble