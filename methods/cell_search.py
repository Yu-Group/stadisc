import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Import fpgrowth methods
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from methods.causal_functions import (get_subgroup_t_statistic, get_subgroup_CATE,
                                      get_subgroup_CATE_std, get_relative_risk,
                                      get_relative_risk_CI)


#============================================================#
##### 1. Choosing features and q values                  #####
#============================================================#

def get_feature_mean_difference(CATE_estimator, q_bot, q_top,
                                features = None):
    """
    Compute the feature mean difference importance scores for a given CATE 
    estimator. Given a subgroup defined using the ITEs of the estimator and 
    quantile thresholds, we compute the means of each feature over this subgroup, 
    and those over the complement of the subgroup, then take their difference,
    and normalize by the sum of the absolute values of all the differences.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    q_bot: float
        q value for bottom quantile cutoff
    q_top: float
        q value for top quantile cutoff
    features: list of strings
        The labels of the features
        
    Returns
    -------
    feature_mean_differences: array-like of shape (n_features,)
        The feature mean difference importance scores
    """
    
    feature_mean_differences = np.zeros(CATE_estimator.X.shape[1])
    for result in CATE_estimator.results.values():
        sg_indicator = result.get_subgroup_indicator(q_bot, q_top, "all")
        X_sg = CATE_estimator.X[sg_indicator,:]
        X_rest = CATE_estimator.X[~sg_indicator,:]
        
        # adding normalized differences
        temp = X_sg.mean(axis = 0) - X_rest.mean(axis = 0)
        feature_mean_differences += temp / np.sum(np.abs(temp))
    feature_mean_differences = feature_mean_differences / CATE_estimator.n_splits
    if features is not None:
        feature_mean_differences = pd.Series(feature_mean_differences, 
                                             index = features)
        
    return feature_mean_differences

def get_subgroup_log_classifier_coef(CATE_estimator, q_bot, q_top,
                                     features = None):
    """
    Compute the logistic classifer coefficient importance scores for a given CATE 
    estimator. Given a subgroup defined using the ITEs of the estimator and 
    quantile thresholds, we use a logistic classifier to predict subgroup membership
    and then output the coefficients of the classifier, normalized by the sum of
    the absolute values of all coefficients.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    q_bot: float
        q value for bottom quantile cutoff
    q_top: float
        q value for top quantile cutoff
    features: list of strings
        The labels of the features
            
    Returns
    -------
    log_classifier_coef: array-like of shape (n_features,)
        The logistic classifer coefficient importance scores
    """    
    
    log_classifier_coef = np.zeros(CATE_estimator.X.shape[1])
    for result in CATE_estimator.results.values():
        sg_indicator = result.get_subgroup_indicator(q_bot, q_top, "all")
        log_classifier = LogisticRegression(penalty = "none", max_iter = 300)
        log_classifier.fit(CATE_estimator.X, sg_indicator)
        
        # log_classifier_coef += (log_classifier.coef_).squeeze()
        # normalization at each fold
        log_classifier_coef += (log_classifier.coef_ / \
                                np.sum(np.abs(log_classifier.coef_))).squeeze()
        
    log_classifier_coef = log_classifier_coef / CATE_estimator.n_splits
    if features is not None:
        log_classifier_coef = pd.Series(log_classifier_coef, index = features)
        
    return log_classifier_coef    

def get_feature_importance_scores(CATE_estimator, kind, q_values, 
                                  features, dir_neg = True):
    """
    Compute feature importance scores for a given CATE estimator, with respect to
    a range of different choices of q values for thresholds.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    kind: string in {"mean_difference", "log_coef"}
        Flag to decide what kind of feature importance score to compute.
    q_values: list
        The q values whose quantiles we will use as thresholds to define the top 
        subgroups.
    features: list of strings
        The labels of the features
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect. Hence, whether lower scores are preferred, 
        or higher scores are preferred.
            
    Returns
    -------
    log_classifier_coef: array-like of shape (n_features,)
        The logistic classifer coefficient importance scores
    """        
    assert kind in ["mean_difference", "log_coef"]
    if kind == "mean_difference":
        get_feat_imp = get_feature_mean_difference
    else:
        get_feat_imp = get_subgroup_log_classifier_coef
    
    feature_importances_df = pd.DataFrame({}, index = features)
    for q in q_values:
        if dir_neg:
            q_bot = 0
            q_top = q
        else:
            q_bot = 1-q
            q_top = 1
        feature_importances_df[q] = get_feat_imp(CATE_estimator, q_bot, q_top)
    feature_importances_max = feature_importances_df.abs().max(axis = 1)
    feature_importances_mean = feature_importances_df.abs().mean(axis = 1)
    feature_importances_df["max"] = feature_importances_max
    feature_importances_df["mean"] = feature_importances_mean
    
    return feature_importances_df

def compare_q_values(CATE_estimator, q_values, dir_neg = True):
    """
    Output a data frame to decide which q values to focus on when doing cell 
    search. For each q value, the data frame includes the mean validation set 
    t-statistic of the subgroup defined by thresholding the ITE predictions 
    using that value, when training the estimator on different training folds, 
    the standard deviation of this statistic across the folds, and also the
    average percentage overlap of the subgroups so derived.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    q_values: list
        The q values whose quantiles we will use as thresholds to define
        the top subgroups.
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
    
    Returns
    -------
    comparison_df: data frame
        A data frame with the t-statistic mean, standard deviation, and 
        average overlap information for each q value.
    """
    
    n_samples = len(CATE_estimator.y)
    n_folds = CATE_estimator.n_splits
    y = CATE_estimator.y
    t = CATE_estimator.t
    comparison_df = pd.DataFrame({})
    for q in q_values:
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
            subgroup_indicator_val = result \
                        .get_subgroup_indicator(q_bot, q_top, "val")
            t_statistics[fold] = get_subgroup_t_statistic(y, t,
                                                    subgroup_indicator_val)
        # Compute mean overlap from the subgroup indicators
        subgroup_indicators = subgroup_indicators.astype(int)
        overlaps_df = (subgroup_indicators.T @ subgroup_indicators) / sg_size
        m = overlaps_df.shape[0]
        overlaps = overlaps_df[np.triu_indices(m, 1)]
        overlap_mean = overlaps.mean()
        t_stat_mean = t_statistics.mean()
        t_stat_std = t_statistics.std() / np.sqrt(n_folds)
         
        comparison_df[q] = pd.Series({"t-stat mean" : t_stat_mean,
                                      "t-stat std" : t_stat_std,
                                      "overlap mean" : overlap_mean})
    comparison_df = comparison_df.T
        
    return comparison_df

#============================================================#
##### 2. Cell search data structure                      #####
#============================================================#

class CellSearch:
    """
    Class for an object to perform one run of the cell search algorithm
    with respect to a CATE estimator fitted on a given training fold. Cells are
    defined to be subgroups defined by constraining the values of a few features.
    The goal is to make the quantile-based top subgroup interpretable by 
    approximating it by a union of a few cells.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    fold: int
        The index of the training fold used to fit the estimator.
    all_features: list of strings
        The labels of all the features in the data set.
    top_features: list of strings
        The labels of the features determined to be most important, and that
        will be used in searching for top cells.
    q: float
        The q values which defines the quantile-based top subgroup of interest.
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect, i.e. whether to select individuals
        below the quantile thresholds or above the quantile thresholds.
    min_support: float
        The min support parameter for FPGrowth, i.e. the algorithm will find all
        "interactions" with support at least this.
    """

    def __init__(self, CATE_estimator, fold, all_features, top_features, q, 
                 dir_neg, min_support = 0.001):
        self.CATE_estimator = CATE_estimator
        self.fold = fold
        if dir_neg:
            q_bot = 0
            q_top = q
        else:
            q_bot = 1-q
            q_top = 1
        # The quantile-based top subgroup is the subgroup defined by selecting 
        # individuals whose predicted ITE is less than (resp. greater than)
        # the q (resp 1-q) quantile of the ITEs
        self.top_quantile_indicator = self.CATE_estimator.results[self.fold] \
                                .get_subgroup_indicator(q_bot, q_top, "all")
        # The cell cover is the list of cells whose union is a good approximation
        # of the quantile-based top subgroup
        self.cell_cover = []
        self.n_samples = len(CATE_estimator.y)
        # The active set is the subset of the samples that lie outside the union
        # of the cells in the cell cover
        self.active_set_indicator = np.array([True] * self.n_samples)
        self.min_support = min_support
        

        self.data_df = pd.DataFrame(CATE_estimator.X, 
                                    columns = all_features)[top_features]
        self.te_df = CellSearch.recode_samples_for_fpgrowth(self.data_df)
        

    def get_true_and_false_positives(self, penalty = 1):
        """
        Produces a data frame which can be used to determine which cells offer
        a good approximation of the quantile-based top subgroup. The returned 
        data frame has columns for the number of true positives (individuals who
        belong to both the cell and the quantile-based top subgroup), the number 
        of false positives (individuals who belong to the cell but not the top 
        quantile subgroup), the number of features used to define the cell,
        the cell size, and the (penalized) difference between the number of
        true positives and the number of false positives.
        
        This function uses FPGrowth in order to run efficiently. Each individual
        (data frame row) is treated as an itemset. FPgrowth finds frequently
        occuring sub-itemsets. In our case, these correspond directly to cells.
        
        Parameters
        ----------
        penalty: float
            A parameter to determine how harshly we want to penalize false 
            positives in computing the tpfp_score. The precise formula is
            tpfp_score = TP - penalty * FP. Hence, a higher penalty prioritizes 
            finding cells that have fewer individuals that do not belong to the 
            quantile-based top subgroup, but may potentially be smaller in size.
        
        Returns
        -------
        cell_ranking_df: data frame
            The data frame containing the information used to rank the cells.
        
        """

        top_quantile_df = self.te_df[self.active_set_indicator & 
                                     self.top_quantile_indicator]
        rest_df = self.te_df[self.active_set_indicator & 
                             ~self.top_quantile_indicator]
        
        # Compute FPGrowth results for both the quantile-based top subgroup and 
        # its complement.
        top_quantile_fpg_df = fpgrowth(top_quantile_df, use_colnames = True, 
                                       min_support = self.min_support)
        rest_fpg_df = fpgrowth(rest_df, use_colnames = True, 
                               min_support = self.min_support)
        
        # Create the data frame used to store the results. We first merge the
        # results of FPGrowth from the previous step. The number of true and
        # false positives can be computed from the support of the cell.
        cell_ranking_df = top_quantile_fpg_df.merge(rest_fpg_df, how = "left",
                                                    on = "itemsets")
        cell_ranking_df.fillna(value = 0, inplace = True)
        cell_ranking_df["true_pos"] = (cell_ranking_df["support_x"] \
                                       * top_quantile_df.shape[0]).astype(int)
        cell_ranking_df["false_pos"] = (cell_ranking_df["support_y"] \
                                        * rest_df.shape[0]).astype(int)
        cell_ranking_df = cell_ranking_df[["itemsets", "true_pos", 
                                           "false_pos"]]
        cell_ranking_df["num_features"] = cell_ranking_df["itemsets"].apply(len)
        cell_ranking_df["cell_size"]  = cell_ranking_df["false_pos"] \
                                        + cell_ranking_df["true_pos"]
        cell_ranking_df["tpfp_score"] = cell_ranking_df["true_pos"] \
                                    - penalty * cell_ranking_df["false_pos"]
        cell_ranking_df.sort_values(by = ["tpfp_score", "num_features"],
                                    ascending = [False, True], inplace = True)
        
        return cell_ranking_df

    def get_top_cells(self, max_features, penalty = 1, tol = 0.05):
        """
        Produce a list of the top cells. These are cells whose tpfp_score lies 
        within a prescribed tolerance level of the best score, and such that
        they are not sub-cells of other cells in the list of top cells.
        
        Parameters
        ----------
        max_features: int
            We restrict ourselves to cells defined by a number of features which
            is at most max_features
        penalty: float
            The penalty parameter for get_true_and_false_positives()
        tol: float
            The tolerance level for defining top cells. The criterion is that their
            tp_fp score must be within false_negatives * tol of the top score, where
            false_negatives is the number of individuals in the quantile-based top 
            subgroup that are not yet in the union of the cell cover.
            
        Returns
        -------
        top_cells: list of sets
            The list of top cells
        """
        cell_ranking_df = self.get_true_and_false_positives(penalty) \
                                        .query(f"num_features <= {max_features}")
        if cell_ranking_df.shape[0] == 0:
            return []
        top_score = cell_ranking_df.iloc[0]["tpfp_score"]
        if top_score < 0:
            return []
        else:
            false_negatives = (self.top_quantile_indicator & 
                               self.active_set_indicator).sum()
            # Also do not select cells with negative tp_fp score.
            threshold = max(top_score - false_negatives * tol, 0)
            top_entries = cell_ranking_df.query(f"tpfp_score >= \
                                                {threshold}")
            top_cells = top_entries["itemsets"]
            # Filtering step: Remove cells that are sub-cells of other top cells
            mask = [True] * len(top_cells)
            for idx, cell in enumerate(top_cells):
                for other_cell in top_cells:
                    if other_cell.issubset(cell) and not cell.issubset(other_cell):
                        mask[idx] = False
            top_cells = top_cells[mask]
                        
            return top_cells

    
    def add_cell(self, itemset):
        """
        Add a cell to the cell cover.
        
        Parameters
        ----------
        itemset: set of strings
            The cell to be added, in itemset form, i.e. represented as a set of 
            strings (one for each feature).
        """
        self.cell_cover.append(itemset)
        cell_indicator = get_cell_indicator(itemset, self.data_df)
        self.active_set_indicator = self.active_set_indicator & (~cell_indicator)

    
    
    def add_k_cells(self, k, max_features, penalty = 1, random = True, 
                    tol = 0.05):
        """
        Add (at most) k cells to the cell cover. For each of k iterations, we check
        if there is at least one top cell that has a non-negative tpfp_score. If so
        we add the cell, otherwise we return.
        
        Parameters
        ----------
        k: int
            The number of cells to be added.
        max_features: int
            We restrict ourselves to cells defined by a number of features which
            is at most max_features
        penalty: float
            The penalty parameter for get_true_and_false_positives()
        random: bool
            Whether or not to randomize the cell to be added at each step. If 
            True, we select a cell uniformly at random from the list of top cells. 
            Otherwise, we select the cell that has the highest tpfp_score.
        tol: float
            The tolerance level for defining top cells. The criterion is that their
            tp_fp score must be within false_negatives * tol of the top score, where
            false_negatives is the number of individuals in the quantile-based top 
            subgroup that are not yet in the union of the cell cover.   
        """
        for _ in range(k):
            top_cells = self.get_top_cells(max_features, penalty, tol)
            if len(top_cells) == 0:
                return None
            elif random:
                top_cell = np.random.choice(top_cells)
                self.add_cell(top_cell)
            else:
                top_cell = list(top_cells)[0]
                self.add_cell(top_cell)
                
    def refresh(self):
        """
        Resets the cell cover and the active set indicator for a fresh cell search
        run.
        """
        self.cell_cover = []
        self.active_set_indicator = np.array([True] * self.n_samples)

    @staticmethod
    def recode_samples_for_fpgrowth(input_df):
        """
        Recode the input data frame into a form that is readable by FPGrowth.

        Parameters
        __________
        input_df : data frame
            The data frame for the study population, using the binary encoding
            used in the original analysis.

        Returns
        _______
        te_df: data frame
            A binary data frame with twice as many columns, i.e. each feature x
            has been split into x_0 and x_1.
        """
        def reformat_row(row):
            # Logic for each row
            return np.array([feature + "_" + str(value) for feature, value 
                             in zip(row.index, row)])

        itemsets = input_df.apply(reformat_row, axis = 1)                
        te = TransactionEncoder()
        te.fit(itemsets)
        te_df = pd.DataFrame(te.transform(itemsets), columns=te.columns_)
        
        return te_df
    

    
#============================================================#
##### 3. Aggregating cell search results                 #####
#============================================================#



def get_cell_search_results(CATE_estimator, all_features, top_features, q_values, 
                            n_cells, max_features, n_reps = 5, penalty = 1, 
                            tol = 0.05, min_support = 0.001, dir_neg = True):
    """
    Run cell search multiple times, n_reps times for every training fold. Collate
    results in a data frame with one column per run and one row for each unique
    cell found during some run.
    
    Parameters
    ----------
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    all_features: list of strings
        The labels of all the features in the data set.
    top_features: list of strings
        The labels of the features determined to be most important, and that
        will be used in searching for top cells.
    q_values: list of float
        The q values whose quantiles we will use as thresholds to define
        the top subgroups.
    n_cells: int
        The number of cells to be added during each run of cell search.
    max_features: int
        We restrict ourselves to cells defined by a number of features which
        is at most max_features
    n_reps: int
        The number of runs for each training fold.
    penalty: float
        The penalty parameter for get_true_and_false_positives()
    tol: float
        The tolerance level for defining top cells. The criterion is that their
        tp_fp score must be within false_negatives * tol of the top score, where
        false_negatives is the number of individuals in the quantile-based top 
        subgroup that are not yet in the union of the cell cover.   
    min_support: float
        The min support parameter for FPGrowth, i.e. the algorithm will find all
        "interactions" with support at least this.
    dir_neg: bool
        A flag on whether we are interested in negative treatment effect
        or positive treatment effect. Hence, whether lower scores are preferred, 
        or higher scores are preferred.
    Returns
    -------
    search_results_df: data frame
        The cell search results
    """
    if n_reps > 1:
        random = True
    else:
        random = False
    search_results_df = pd.DataFrame({"cells" : []})
    for fold in tqdm(range(CATE_estimator.n_splits)):
        for q in q_values:
            cell_search_run = CellSearch(CATE_estimator, fold, all_features, 
                                         top_features, q, dir_neg, min_support)
            for b in range(n_reps):
                cell_search_run.refresh()
                cell_search_run.add_k_cells(n_cells, max_features, penalty, 
                                            random, tol)
                run_results = pd.DataFrame({"cells" : cell_search_run.cell_cover, 
                                            f"q={q}/fold={fold}/b={b}" : 1})
                search_results_df = search_results_df.merge(run_results, 
                                                how = "outer", on = "cells")
            del cell_search_run
    search_results_df = search_results_df.set_index("cells").fillna(0) \
                                                            .astype(int)
    
    return search_results_df
    
def aggregate_cell_search_results(search_results_df, data_df, 
                                  collapse_cells = False):
    """
    Aggregate the cell search results reported by get_cell_search_results(). The
    output is a data frame that has one column for each q value investigated in
    search_results_df. Each row corresponds to a cell, and each entry is the number
    of times that cell was selected during a cell search run for a particular choice
    of q. If the collapse_cells flag is set to True, then to the count of each cell,
    we also add the count of sub-cells that occur, weighted by their size (in terms 
    of number of samples) relative to that of the super-cell.

    Parameters
    ----------
    search_results_df: data frame
        The data frame returned by a call to get_cell_search_results().
    data_df: data frame
        The data frame containing the covariate data.
    collapse_cells: bool
        Flag for whether or not to collapse the cells.

    Returns
    -------
    aggregated_df: data frame
        A data frame containing the aggregated results.
    """
    aggregated_df = pd.DataFrame({"cells" : search_results_df.index})
    q_values = list({float(item.split("/")[0][2:]) for item in 
                     search_results_df.columns})
    q_values = sorted(q_values)
    for q in q_values:
        cell_counts = search_results_df.filter(regex=str(q)).sum(axis = 1)
        cell_counts = cell_counts[cell_counts > 0].astype(float)
        if collapse_cells:
            collapsed_cell_counts = pd.Series(np.zeros_like(cell_counts),
                                              index = cell_counts.index)
            cell_list = list(cell_counts.index)
            for cell in cell_list:
                cell_indicator = get_cell_indicator(cell, data_df)
                for other_cell in cell_list:
                    if cell.issubset(other_cell):
                        other_cell_indicator = get_cell_indicator(other_cell, data_df)
                        weight = other_cell_indicator.sum() / cell_indicator.sum()
                        collapsed_cell_counts[cell] += cell_counts[other_cell] * weight
            cell_counts = collapsed_cell_counts
        q_results = pd.DataFrame({"cells" : cell_counts.index,
                                  f"q={q}" : cell_counts.values})
        aggregated_df = aggregated_df.merge(q_results, how = "outer", on = "cells")
    aggregated_df = aggregated_df.set_index("cells").fillna(0)
    num_runs = search_results_df.shape[1] // len(q_values)
    total_freq = aggregated_df.sum(axis = 1) / (num_runs * aggregated_df.shape[1])
    max_freq = aggregated_df.max(axis = 1) / num_runs
    aggregated_df["num runs"] = num_runs
    aggregated_df["total frequency"] = total_freq
    aggregated_df["max frequency"] = max_freq
    aggregated_df = aggregated_df.sort_values("total frequency", ascending = False)

    return aggregated_df

def get_coverage_results(cell_list, data_df, CATE_estimator, q_values, dir_neg):
    """
    Compute statistics on how well a cell cover approximates the quantile-based 
    top subgroup of a CATE estimator with respect to a range of q values. We 
    compute the number of true positives (TP), false positives (FP), and false 
    negatives (FN) for the quantile-based top subgroup for the CATE estimator 
    trained on each fold, and report their mean and standard deviation. We do this 
    for each q value requested.
    
    Parameters
    ----------
    cell_list: list of sets
        The list of cells in the cell cover.
    data_df: data frame
        The data frame containing the covariate data.
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    q_values: list of float
        The q values whose quantiles we will use as thresholds to define
        the top subgroups.
    dir_neg: bool
        
    
    Returns
    -------
    coverage_results_df: data frame
        Data frame containing the coverage results.
    """
    
    n_samples = data_df.shape[0]
    cell_cover_indicator = np.array([False] * n_samples)
    for cell in cell_list:
        cell_indicator = get_cell_indicator(cell, data_df)
        cell_cover_indicator = cell_cover_indicator | cell_indicator
    coverage_results_df = pd.DataFrame({}, index = ["TP mean", "FP mean",
                                                    "FN mean", "TP std",
                                                    "FN std"])
    for q in q_values:
        if dir_neg:
            q_bot = 0
            q_top = q
        else:
            q_bot = 1-q
            q_top = 1
        TP_scores = []
        FP_scores = []
        FN_scores = []
        for result in CATE_estimator.results.values():
            top_quantile_indicator = result \
                                     .get_subgroup_indicator(q_bot, q_top, "all")
            TP = (top_quantile_indicator & cell_cover_indicator).sum()
            FP = (~top_quantile_indicator & cell_cover_indicator).sum()
            FN = (top_quantile_indicator & ~cell_cover_indicator).sum()
            TP_scores.append(TP)
            FP_scores.append(FP)
            FN_scores.append(FN)
        TP_scores = np.array(TP_scores)
        FP_scores = np.array(FP_scores)
        FN_scores = np.array(FN_scores)
        TP_mean = TP_scores.mean()
        FP_mean = FP_scores.mean()
        FN_mean = FN_scores.mean()
        TP_std = TP_scores.std()
        FN_std = FN_scores.std()
        coverage_results_df[q] = [TP_mean, FP_mean, FN_mean, TP_std, FN_std]
    coverage_results_df = coverage_results_df.T

    return coverage_results_df

def get_cell_overlap_results(cell_list, data_df, score_type):
    """
    Compute the pairwise overlap values for cells in the cell list (with 
    reference) to the sample population whose covariate information is stored 
    in data_df. Overlap can be measured in one of two ways, either using the 
    size of intersection between the two groups, or using the cosine similarity 
    of their indicator vectors (this normalizes the intersection size by the 
    geometric mean of the sizes of the two cells.)
    
    Parameters
    ----------
    cell_list: list of sets
        The list of cells in the cell cover.
    data_df: data frame
        The data frame containing the covariate data.
    score_type: string in {"intersection_size", "cosine_sim"}
        Flag to indicate what overlap measure to use.
        
    Returns
    -------
    overlap_mat: array-like of (n_cells, n_cells)
        The matrix of pairwise overlap information.
    """
    assert score_type in {"intersection_size", "cosine_sim"}
    indicator_matrix = pd.DataFrame({})
    n_samples = data_df.shape[0]
    for idx, cell in enumerate(cell_list):
        cell_indicator = get_cell_indicator(cell, data_df)
        indicator_matrix[f"Cell {idx+1}"] = cell_indicator.astype(int)
    if score_type == "intersection_size":
        overlap_mat = indicator_matrix.T @ indicator_matrix
    elif score_type == "cosine_sim":
        overlap_mat = pd.DataFrame(cosine_similarity(indicator_matrix.T),
                                   columns = indicator_matrix.columns,
                                   index = indicator_matrix.columns)
    
    return overlap_mat

def get_cell_significance_results(cell_list, y, t, data_df, CATE_estimator = None,
                                  on_trainval = True):
    """
    Compute the CATE and t-statistic values for each cell in the cell list.
    
    Parameters
    ----------
    cell_list: list of sets
        The list of cells in the cell cover.
    y: array-like of shape (n_samples,)
        Vector of observed responses for entire training-validation set.
    t: array-like of shape (n_samples,)
        Vector of treatment assignments for entire training-validation set.
    data_df: data frame
        The data frame containing the covariate data for the entire training
        -validation set.
    CATE_estimator: CATEEstimatorWrapper
        Fitted CATE estimator to be used to predict ITEs.
    on_trainval: bool
        Flag for whether the data used is the trainval data set. If True, two kinds 
        of t-statistics are produced: One computed on the entire data set, and the
        mean and standard deviation of those computed on training folds of different
        splits.
    
    Returns
    -------
    cell_significance_df: data frame
        Data frame containing the results.
    """
    
    cell_significance_df = pd.DataFrame({})
    n_samples = len(y)
    union_indicator = np.array([False] * n_samples)
    
    def get_subgroup_results(indicator):
        size = indicator.sum()
        events = y[indicator].sum()
        CATE = get_subgroup_CATE(y, t, indicator)
        CATE_std = get_subgroup_CATE_std(y, t, indicator)
        overall_t_stat = get_subgroup_t_statistic(y, t, indicator)
        cell_entry = [size, events, CATE, CATE_std, overall_t_stat]
        if on_trainval:
            val_t_stat_values = []
            for result in CATE_estimator.results.values():
                val_t_stat = get_subgroup_t_statistic(y, t, indicator,
                                                      result.val_indicator)
                val_t_stat_values.append(val_t_stat)
            val_t_stat_values = np.array(val_t_stat_values)
            val_t_stat_mean = val_t_stat_values.mean()
            val_t_stat_std = val_t_stat_values.std() / \
                             np.sqrt(len(val_t_stat_values))
            cell_entry += [val_t_stat_mean, val_t_stat_std]
        return cell_entry
    
    # Get results for individual cells
    for cell in cell_list:
        cell_name = recode_itemset_into_query(cell)
        cell_indicator = get_cell_indicator(cell, data_df)
        union_indicator = union_indicator | cell_indicator
        cell_significance_df[cell_name] = get_subgroup_results(cell_indicator)
                                           
    # Get results for union
    cell_significance_df["union"] = get_subgroup_results(union_indicator)
    
    cell_significance_df = cell_significance_df.T
    columns = ["size", "num_evts", "CATE", "CATE_std", "t-stat (overall)"]
    if on_trainval:
        columns += ["t-stat (val) mean", "t-stat (val) std"]
    cell_significance_df.columns = columns
    cell_significance_df["size"] = cell_significance_df["size"].astype(int)
    cell_significance_df["num_evts"] = cell_significance_df["num_evts"].astype(int)
    
    return cell_significance_df

def get_RR_results(cell_list, y, t, data_df):
    """
    Compute relative risk and confidence intervals for cells in the cell list.
    
    Parameters
    ----------
    cell_list: list of sets
        The list of cells in the cell cover.
    y: array-like of shape (n_samples,)
        Vector of observed responses for entire training-validation set.
    t: array-like of shape (n_samples,)
        Vector of treatment assignments for entire training-validation set.
    data_df: data frame
        The data frame containing the covariate data for the entire training
        -validation set.

    Returns
    -------
    RR_results_df: data frame
        Data frame containing the results.
    """
    RR_results_df = pd.DataFrame({}, 
        index = ["size", "relative risk", "CI lower endpoint", 
                 "CI upper endpoint"])
    n_samples = len(y)
    union_indicator = np.array([False] * n_samples)
    # Get results for individual cells
    for cell in cell_list:
        cell_name = recode_itemset_into_query(cell)
        cell_indicator = get_cell_indicator(cell, data_df)
        union_indicator = union_indicator | cell_indicator
        size = cell_indicator.sum()
        y_g = y[cell_indicator]
        t_g = t[cell_indicator]
        RR = get_relative_risk(y_g, t_g)
        RR_CI = get_relative_risk_CI(y_g, t_g)
        RR_results_df[cell_name] = np.array([size, RR, RR_CI[0], RR_CI[1]])
    # Get results for union
    y_g = y[union_indicator]
    t_g = t[union_indicator]
    RR = get_relative_risk(y_g, t_g)
    RR_CI = get_relative_risk_CI(y_g, t_g)
    size = len(y_g)
    RR_results_df["union"] = np.array([size, RR, RR_CI[0], RR_CI[1]])
    # Get results for the entire population
    RR = get_relative_risk(y, t)
    RR_CI = get_relative_risk_CI(y, t)
    size = len(y)
    RR_results_df["entire population"] = np.array([size, RR, RR_CI[0], RR_CI[1]])
    
    RR_results_df = RR_results_df.T
    RR_results_df["size"] = RR_results_df["size"].astype(int)
    
    return RR_results_df

#============================================================#
##### 4. Miscellaneous                                   #####
#============================================================#


def recode_itemset_into_query(itemset):
    """
    Recode a cell in itemset form (list of strings), into a query form (a
    string readable by the query method of a pandas data frame.)

    Parameters
    ----------
    itemset: list of strings
        The cell to be added, in itemset form, i.e. represented as a set of 
        strings (one for each feature).

    Returns
    -------
    query_string: string
        A string readable by the query method of a pandas data frame, such 
        that the result of this query on the original data frame returns the 
        subset of the data belonging to the cell.
    """
    query_string = ""
    for item in itemset:
        feature = item[:-2]
        level = item[-1]
        if len(query_string) > 0:
            query_string = query_string + " & " + feature + "==" + level
        else:
            query_string = feature + "==" + level

    return query_string

def get_cell_indicator(itemset, data_df):
    """
    Get the indicator for a cell coded as an itemset, with respect to the
    population in data_df.
    
    Parameters
    ----------
    itemset: list of strings
        The cell whose indicator we want.
    data_df: data frame
        The data frame containing the covariate data.

    Returns
    -------
    cell_indicator: array-like of shape (n_samples,)
        The cell indicator.
    """
    
    query_string = recode_itemset_into_query(itemset)
    cell_indices = data_df.query(query_string).index
    cell_indicator = np.array([i in cell_indices for i in 
                               range(data_df.shape[0])])
    
    return cell_indicator