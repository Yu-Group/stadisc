import numpy as np
import pandas as pd
import copy
import random

from sklearn.base import is_regressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

from causalml.inference.meta import (BaseSClassifier, BaseRClassifier, 
                                     BaseTClassifier, BaseXClassifier,
                                     BaseSRegressor, BaseRRegressor, 
                                     BaseTRegressor, BaseXRegressor)
from causalml.inference.tree import CausalTreeRegressor
from methods.causal_forest import CausalForest

class CATEEstimatorResults:
    """
    A container for the results for a CATE estimator when trained on subset of 
    the training-validation set.
    
    Parameters
    ----------
    train_indices: array-like of shape (n_samples_train,)
                   Vector of indices of samples in training set.
    val_indices: array-like of shape (n_samples_train,)
                 Vector of indices of samples in validation set.
    CATE_estimator: CATEEstimatorWrapper object
                    The CATEEstimatorWrapper object whose results are being stored.
    save_metalearner: bool
                      A flag for whether to save a copy of the metalearner.
    """
    
    def __init__(self, train_indices, val_indices, CATE_estimator, 
                 save_metalearner = False):
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        if save_metalearner:
            self.meta_learner = copy.deepcopy(CATE_estimator.meta_learner)
        else:
            self.meta_learner = CATE_estimator.meta_learner
        self._fit_meta_learner(CATE_estimator)
        self._predict_tau(CATE_estimator)

    def _fit_meta_learner(self, CATE_estimator):
        """
        Helper function to fit the metalearner on the training set.       
        """
        
        # Create training set
        X_train = CATE_estimator.X[self.train_indices]
        y_train = CATE_estimator.y[self.train_indices]
        t_train = CATE_estimator.t[self.train_indices]
        
        # Set random seed for reproducible results.
        np.random.seed(405)
        random.seed(405)
        
        # Different metalearners have different signatures for their fit method
        if isinstance(CATE_estimator, (XLearnerWrapper, RLearnerWrapper)):
            p = 0.5*np.ones_like(t_train) # Random experiment, so propensity score = 0.5
            self.meta_learner.fit(X_train, t_train, y_train, p)
        elif isinstance(CATE_estimator, CausalForestWrapper):
            self.meta_learner.fit(X_train, y_train, t_train)
        else:
            self.meta_learner.fit(X_train, t_train, y_train)
            
    def _predict_tau(self, CATE_estimator):
        """
        Helper function to predict individual treatment effects for all individuals
        in training-validation set.
        """
        X = CATE_estimator.X
        # Different metalearners have different signatures for their predict method
        if isinstance(CATE_estimator, XLearnerWrapper):
            self.tau = self.meta_learner.predict(X, p = 0.5*np.ones(X.shape[0])).squeeze()
        else:
            self.tau = self.meta_learner.predict(X).squeeze()
            
    @property        
    def tau_train(self):
        """
        Function that returns ITE predictions in training set.
        """
        return self.tau[self.train_indices]
    
    @property        
    def tau_val(self):
        """
        Function that returns ITE predictions in validation set.
        """
        return self.tau[self.val_indices]
        
    def get_subgroup_indicator(self, q_bot, q_top, kind = "val"):
        """
        Compute the indices of the samples whose estimated ITE lies between two
        quantile values

        Parameters
        ----------
        q_bot: float
               q value for bottom quantile cutoff
        q_top: float
               q value for top quantile cutoff
        kind: string in {"val", "train", "all"}
              Flag to indicate whether to restrict to samples in the validation set, 
              training set, or the entire training-validation set.

        Returns
        -------
        subgroup_indices: array-like
                          Vector of indices of samples in subgroup
        """
        
        # Check that arguments have the right form
        assert 0 <= q_bot and q_bot < q_top and q_top <= 1
        assert kind in ["val", "train", "all"]
        
        # Compute top and bottom quantile cutoffs
        quantile_bot = np.quantile(self.tau_train, q_bot)
        quantile_top = np.quantile(self.tau_train, q_top)
        
        # Get subgroup indicator for entire training-validation set
        if q_bot == 0:
            subgroup_indicator = (self.tau <= quantile_top)
        else:
            subgroup_indicator = (quantile_bot < self.tau) & \
                                        (self.tau <= quantile_top)
        
        # Get subgroup indices of the desired kind
        n_samples = len(subgroup_indicator)
        if kind == "val":
            valset_indicator = np.array([idx in self.val_indices for 
                                         idx in range(n_samples)])
            subgroup_indicator = subgroup_indicator & valset_indicator
        elif kind == "train":
            trainset_indicator = np.array([idx in self.train_indices for 
                                           idx in range(n_samples)])
            subgroup_indicator = subgroup_indicator & trainset_indicator
        else:
            pass
        
        return subgroup_indicator
    
    @property
    def val_indicator(self):
        """
        Function that returns an indicator of the validation set
        """
        return np.array([i in self.val_indices for i in range(len(self.tau))])
    
    @property
    def train_indicator(self):
        """
        Function that returns an indicator of the training set
        """
        return np.array([i in self.train_indices for i in range(len(self.tau))])
    
        
class BaseCATEEstimatorWrapper:
    """
    Base class for all CATE estimator wrapper objects. These objects provide a 
    convenient interface for tuning the hyperparameters for base learners of 
    metalearners, as well as for obtaining individual treatment effect predictions
    of the methods when trained using different training-validation splits.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
       Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
       Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    """
    def __init__(self, X, t, y, cv):

        self.X = X
        self.t = t
        self.y = y
        self.cv = cv
        self.n_splits = self.cv.get_n_splits()
        self.results = {}
        self.meta_learner = None
        self.fitted = False

    def tune_params(self, n_iter, verbose = 0):
        """
        Tune hyperparameters for the base learners. Implemented differently for each 
        metalearner type (implemented as a subclass.)
        
        Parameters
        ----------
        n_iter: int
                Number of iterations for RandomizedSearchCV used to tune parameters
        verbose: int
                 verbose argument for RandomizedSearchCV
        """
        pass
    
    def set_params(self, params):
        """
        Set hyperparameters for the base learners. Implemented differently for each 
        metalearner type (implemented as a subclass.)
        
        Parameters
        ----------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners
        """        
        pass
    
    def get_params(self):
        """
        Get hyperparameters for the base learners. Implemented differently for each 
        metalearner type (implemented as a subclass.)
        
        Returns
        -------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners
        """
        pass        
    
    def transfer_params(self, other):
        """
        Adopt hyperparameters of another CATEEstimatorWrapper object. Implemented 
        differently for each metalearner type.
        
        Parameters
        ----------
        other: CATEEstimatorWrapper
               Another wrapper object from which to obtain hyperparameters
        """          
        self.set_params(other.get_params())
        
        
    def fit(self):
        """
        Using each split from K fold CV, fit the estimator on the training set, and 
        obtain ITE predictions for all individuals in the data set. Store these in a 
        CATEEstimatorResults object.
        """
        for idx, (train_indices, val_indices) in \
                enumerate(self.cv.split(self.X, self.y + 2*self.t)):
            self.results[idx] = CATEEstimatorResults(train_indices, val_indices, self)
        self.fitted = True
        
    def __repr__(self):
        return "Wrapper for " + self.meta_learner.__repr__()

class SLearnerWrapper(BaseCATEEstimatorWrapper):
    """
    Class for a wrapper around S Learner metalearner.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
       Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
       Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    base_learner: An sklearn estimator object
    param_grid: dict
                dict of grids for hyperparameter search
    params: dict
            dict of hyperparameters for base learner
    """
    def __init__(self, X, t, y, cv, base_learner, param_grid = {}, params = {}):
        
        super().__init__(X, t, y, cv)
        self.base_learner = copy.deepcopy(base_learner).set_params(**params)
        self.param_grid = param_grid
        
        # Check if base learner is regressor or classifier, and initialize the 
        # appropriate metalearner object
        if is_regressor(self.base_learner):
            self.meta_learner_class = BaseSRegressor
            self.scoring = "r2"
        else:
            self.meta_learner_class = BaseSClassifier
            self.scoring = "neg_log_loss"
        self.meta_learner = self.meta_learner_class(learner = self.base_learner)
    
    def tune_params(self, n_iter = 10, verbose = 0):
        """
        Tune hyperparameters for the base learner.
        
        Parameters
        ----------
        n_iter: int
                Number of iterations for RandomizedSearchCV used to tune parameters
        verbose: int
                 verbose argument for RandomizedSearchCV
        """
        
        Xt = np.hstack((self.X, self.t.reshape(-1,1)))
        rand_search = RandomizedSearchCV(self.base_learner, self.param_grid,
                                         cv = self.cv.split(self.X, self.y + 2*self.t), 
                                         n_iter = n_iter, scoring = self.scoring, 
                                         verbose = verbose)
        rand_search.fit(Xt, self.y)
        self.base_learner = rand_search.best_estimator_
        self.meta_learner = self.meta_learner_class(learner = self.base_learner)
        self.base_learner_score = rand_search.best_score_
    
    def set_params(self, params):
        """
        Set hyperparameters for the base learner.
        
        Parameters
        ----------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learner.
        """   
        self.base_learner.set_params(**params["base_learner"])
    
    def get_params(self):
        """
        Get hyperparameters for the base learner.
        
        Returns
        -------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learner.
        """
        params = {"base_learner" : self.base_learner.get_params()}
        
        return params

class TLearnerWrapper(BaseCATEEstimatorWrapper):
    """
    Class for a wrapper around T Learner metalearner.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
       Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
       Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    base_learner: An sklearn estimator object
    param_grid: dict
                Dictionary of grids for hyperparameter search for base learner
    params_treat: dict
                  Dictionary of hyperparameters for outcome model under treatment
    params_control: dict
                    Dictionary of hyperparameters for outcome model under control
    """    
    
    def __init__(self, X, t, y, cv, base_learner, param_grid = {}, 
                 params_treat = {}, params_control = {}):
        
        super().__init__(X, t, y, cv)
        self.treatment_outcome_learner = copy.deepcopy(base_learner) \
                                            .set_params(**params_treat)
        self.control_outcome_learner = copy.deepcopy(base_learner) \
                                           .set_params(**params_control)
        self.param_grid = param_grid
        
        # Check if base learner is regressor or classifier, and initialize the 
        # appropriate metalearner object
        if is_regressor(self.treatment_outcome_learner):
            self.meta_learner_class = BaseTRegressor
            self.scoring = "r2"
        else:
            self.meta_learner_class = BaseTClassifier
            self.scoring = "neg_log_loss"
        self._make_meta_learner()
             
    def _make_meta_learner(self):
        """
        Helper function that initializes the metalearner object
        """
        self.meta_learner = self.meta_learner_class(control_learner = 
                                                    self.control_outcome_learner,
                                                    treatment_learner = 
                                                    self.treatment_outcome_learner)
        
    def tune_params(self, n_iter = 10, verbose = 0):
        """
        Tune hyperparameters for the base learner.
        
        Parameters
        ----------
        n_iter: int
                Number of iterations for RandomizedSearchCV used to tune parameters
        verbose: int
                 verbose argument for RandomizedSearchCV
        """        
        X_treat = self.X[self.t == 1]
        y_treat = self.y[self.t == 1]
        X_control = self.X[self.t == 0]
        y_control = self.y[self.t == 0]
        rand_search_treat = RandomizedSearchCV(self.treatment_outcome_learner, 
                                               self.param_grid, 
                                               cv = self.cv.split(X_treat, y_treat), 
                                               n_iter = n_iter, scoring = self.scoring, 
                                               verbose = verbose)
        rand_search_treat.fit(X_treat, y_treat)
        self.treatment_outcome_learner = rand_search_treat.best_estimator_
        
        rand_search_control = RandomizedSearchCV(self.control_outcome_learner, 
                                                 self.param_grid,
                                                 cv = self.cv.split(X_control, y_control), 
                                                 n_iter = n_iter, scoring = self.scoring, 
                                                 verbose = verbose)
        rand_search_control.fit(X_control, y_control)
        self.control_outcome_learner = rand_search_control.best_estimator_
        self._make_meta_learner()
        self.base_learner_scores = (rand_search_treat.best_score_, 
                                    rand_search_control.best_score_)
    
    def set_params(self, params):
        """
        Set hyperparameters for the base learners.
        
        Parameters
        ----------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners.
        """   
        self.control_outcome_learner \
                        .set_params(**params["control_outcome_learner"])
        self.treatment_outcome_learner \
                        .set_params(**params["treatment_outcome_learner"])
    
    def get_params(self):
        """
        Get hyperparameters for the base learners. Implemented differently for each 
        metalearner type (implemented as a subclass.)
        
        Returns
        -------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners.
        """
        return {"control_outcome_learner" : self.control_outcome_learner \
                                                                .get_params(),
                "treatment_outcome_learner" : self.treatment_outcome_learner \
                                                                .get_params()}
            
class XLearnerWrapper(BaseCATEEstimatorWrapper):
    """
    Class for a wrapper around X Learner metalearner.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
       Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
       Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    outcome_learner: sklearn estimator object
                     sklearn estimator for the outcome 
    effect_learner: sklearn estimator object
                    sklearn estimator for the treatment effect
    outcome_param_grid: dict
                        Dictionary of grids for hyperparameter search for 
                        outcome estimator
    effect_param_grid: dict
                       Dictionary of grids for hyperparameter search for 
                       effect estimator
    params_treat: dict
                  Dictionary of hyperparameters for outcome model under treatment
    params_control: dict
                    Dictionary of hyperparameters for outcome model under control
    params_treat_effect: dict
                  Dictionary of hyperparameters for outcome model under treatment
    params_control_effect: dict
                    Dictionary of hyperparameters for outcome model under control                
    """    
    
    def __init__(self, X, t, y, cv, outcome_learner, effect_learner, 
                 outcome_param_grid = {}, effect_param_grid = {},
                 params_treat = {}, params_control = {},
                 params_treat_effect = {}, params_control_effect = {}):
        
        super().__init__(X, t, y, cv)
        self.treatment_outcome_learner = copy.deepcopy(outcome_learner) \
                                            .set_params(**params_treat)
        self.control_outcome_learner = copy.deepcopy(outcome_learner) \
                                        .set_params(**params_control)
        self.outcome_param_grid = outcome_param_grid  # param grid for effect learner
        self.treatment_effect_learner = copy.deepcopy(effect_learner) \
                                            .set_params(**params_treat_effect)
        self.control_effect_learner = copy.deepcopy(effect_learner) \
                                            .set_params(**params_control_effect)
        self.effect_param_grid = effect_param_grid # param grid for effect learner
        # Flag to tune the outcome learners. Set to True if params_treat or 
        # params_control is the empty dictionary. The outcome learners are tuned
        # exactly the same way as those for a T-learner, so we don't have to
        # repeat this part of the tuning.
        self.tune_outcome_learners = (len(params_treat) == 0) or \
                                            (len(params_control) == 0)
        
        # Check if outcome learner is regressor or classifier, and initialize the 
        # appropriate metalearner object
        if is_regressor(self.treatment_outcome_learner):
            self.meta_learner_class = BaseXRegressor
            self.scoring = "r2"
        else:
            self.meta_learner_class = BaseXClassifier
            self.scoring = "neg_log_loss"
        self._make_meta_learner()
    
    def _make_meta_learner(self):
        """
        Helper function that initializes the metalearner object
        """
        self.meta_learner = self.meta_learner_class(control_outcome_learner = 
                                                    self.control_outcome_learner,
                                                    treatment_outcome_learner = 
                                                    self.treatment_outcome_learner,
                                                    control_effect_learner = 
                                                    self.control_effect_learner,
                                                    treatment_effect_learner = 
                                                    self.treatment_effect_learner)
    
    def tune_params(self, n_iter = 10, verbose = 0):
        """
        Tune hyperparameters for the base learner.
        
        Parameters
        ----------
        n_iter: int
                Number of iterations for RandomizedSearchCV used to tune parameters
        verbose: int
                 verbose argument for RandomizedSearchCV
        """ 
        X_treat = self.X[self.t == 1]
        y_treat = self.y[self.t == 1]
        X_control = self.X[self.t == 0]
        y_control = self.y[self.t == 0]
        
        if self.tune_outcome_learners:
            # Tune the outcome estimators if they are not set
            rand_search_treat = RandomizedSearchCV(self.treatment_outcome_learner,
                                                   self.outcome_param_grid, 
                                                   cv = self.cv.split(X_treat, 
                                                                      y_treat),
                                                   n_iter = n_iter, 
                                                   scoring = self.scoring,
                                                   verbose = verbose)
            rand_search_treat.fit(X_treat, y_treat)
            self.treatment_outcome_learner = rand_search_treat.best_estimator_

            rand_search_control = RandomizedSearchCV(self.control_outcome_learner, 
                                                     self.outcome_param_grid,
                                                     cv = self.cv.split(X_control, 
                                                                        y_control), 
                                                     n_iter = n_iter, 
                                                     scoring = self.scoring,
                                                     verbose = verbose)
            rand_search_control.fit(X_control, y_control)
            self.control_outcome_learner = rand_search_control.best_estimator_
            self.base_learner_scores = (rand_search_treat.best_score_, 
                                        rand_search_control.best_score_)
        else:
            pass
        
        # Tune the treatment effect estimators: We first impute the missing outcomes
        # using the fitted outcome learners, and use this to impute a treatment 
        # effect for every individual. We then fit the treatment effect learners 
        # using the imputed treatment effect as the target. We tune the hyperparameters 
        # for the treatment effect models this way.
        self.treatment_outcome_learner.fit(X_treat, y_treat)
        self.control_outcome_learner.fit(X_control, y_control)
        y0_treat_ = self.control_outcome_learner.predict(X_treat)
        y1_control_ = self.treatment_outcome_learner.predict(X_control)
        rand_search_treat_effect = RandomizedSearchCV(self.treatment_effect_learner, 
                                                      self.effect_param_grid, 
                                                      cv = self.cv.split(X_treat, 
                                                                         y_treat), 
                                                      n_iter = n_iter)
        rand_search_treat_effect.fit(X_treat, y_treat - y0_treat_)
        self.treatment_effect_learner = rand_search_treat_effect.best_estimator_
        
        rand_search_control_effect = RandomizedSearchCV(self.control_effect_learner, 
                                                        self.effect_param_grid, 
                                                        cv = self.cv.split(X_control, 
                                                                           y_control), 
                                                        n_iter = n_iter)
        rand_search_control_effect.fit(X_control, y1_control_ - y_control)
        self.control_effect_learner = rand_search_control_effect.best_estimator_
        self.effect_learner_scores = (rand_search_treat_effect.best_score_, 
                                      rand_search_control_effect.best_score_)
        self._make_meta_learner()
    
    def set_params(self, params):
        """
        Set hyperparameters for the base learners.
        
        Parameters
        ----------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners.
        """  
        self.control_outcome_learner.set_params(**params["control_outcome_learner"])
        self.treatment_outcome_learner.set_params(**params["treatment_outcome_learner"])
        self.control_effect_learner.set_params(**params["control_effect_learner"])
        self.treatment_effect_learner.set_params(**params["treatment_effect_learner"])
        
    def get_params(self):
        """
        Get hyperparameters for the base learners. Implemented differently for each 
        metalearner type (implemented as a subclass.)
        
        Returns
        -------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners.
        """
        
        return {"control_outcome_learner" : self.control_outcome_learner.get_params(),
                "treatment_outcome_learner" : self.treatment_outcome_learner.get_params(),
                "control_effect_learner" : self.control_effect_learner.get_params(),
                "treatment_effect_learner" : self.treatment_effect_learner.get_params()}
    
class RLearnerWrapper(BaseCATEEstimatorWrapper):
    """
    Class for a wrapper around X Learner metalearner.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
       Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
       Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    outcome_learner: sklearn estimator object
                     sklearn estimator for the outcome model
    effect_learner: sklearn estimator object
                    sklearn estimator for the treatment effect model
    outcome_param_grid: dict
                        Dictionary of grids for hyperparameter search for 
                        outcome estimator
    effect_param_grid: dict
                       Dictionary of grids for hyperparameter search for 
                       effect estimator
    params_outcome: dict
                    Dictionary of hyperparameters for outcome model under control
    params_effect: dict
                   Dictionary of hyperparameters for outcome model under treatment               
    """   
    
    def __init__(self, X, t, y, cv, outcome_learner, effect_learner, 
                 outcome_param_grid = {}, effect_param_grid = {}, 
                 params_outcome = {}, params_effect = {}):
        
        super().__init__(X, t, y, cv)
        self.outcome_learner = copy.deepcopy(outcome_learner) \
                                    .set_params(**params_outcome)
        self.outcome_param_grid = outcome_param_grid
        self.effect_learner = copy.deepcopy(effect_learner) \
                                .set_params(**params_effect)
        self.effect_param_grid = effect_param_grid
        
        # Check if outcome learner is regressor or classifier, and initialize the 
        # appropriate metalearner object
        if is_regressor(self.outcome_learner):
            self.meta_learner_class = BaseRRegressor
        else:
            self.meta_learner_class = BaseRClassifier
        self._make_meta_learner()

    def _make_meta_learner(self):
        """
        Helper function that initializes the metalearner object
        """
        self.meta_learner = self.meta_learner_class(outcome_learner = 
                                                    self.outcome_learner, 
                                                    effect_learner = 
                                                    self.effect_learner)
            
    def tune_params(self, n_iter = 10, verbose = 0):
        """
        Tune hyperparameters for the base learner.
        
        Parameters
        ----------
        n_iter: int
                Number of iterations for RandomizedSearchCV used to tune 
                parameters
        verbose: int
                 verbose argument for RandomizedSearchCV
        """ 
        rand_search_outcome = RandomizedSearchCV(self.outcome_learner, 
                                  self.outcome_param_grid, 
                                  cv = self.cv.split(self.X, self.y + 2*self.t), 
                                  n_iter = n_iter, verbose = verbose)
        rand_search_outcome.fit(self.X, self.y)
        self.outcome_learner = rand_search_outcome.best_estimator_
        
        y_hat = cross_val_predict(self.outcome_learner, self.X, self.y, 
                                  cv = self.cv.split(self.X, self.y + 2*self.t))
        rand_search_effect = RandomizedSearchCV(self.effect_learner, 
                                 self.effect_param_grid, 
                                 cv = self.cv.split(self.X, self.y + 2*self.t), 
                                 n_iter = n_iter, verbose = verbose)
        rand_search_effect.fit(self.X, (self.y - y_hat) / (self.t - 0.5))
        self.effect_learner = rand_search_effect.best_estimator_
        self.learner_scores = (rand_search_outcome.best_score_, 
                               rand_search_effect.best_score_)
        self._make_meta_learner()
        
    def set_params(self, params):
        """
        Set hyperparameters for the base learners.
        
        Parameters
        ----------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners.
        """  
        self.outcome_learner.set_params(**params["outcome_learner"])
        self.effect_learner.set_params(**params["effect_learner"])
        
    def get_params(self):
        """
        Get hyperparameters for the base learners. Implemented differently 
        for each metalearner type (implemented as a subclass.)
        
        Returns
        -------
        params: dictionary of dictionaries
                Dictionary of hyperparameter dictionaries for base learners.
        """
        return {"outcome_learner" : self.outcome_learner.get_params(),
                "effect_learner" : self.effect_learner.get_params()}
    

class CausalTreeWrapper(BaseCATEEstimatorWrapper):
    """
    Class for a wrapper around the Causal Tree estimator.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
       Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
       Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    params: dict
            Dictionary of hyperparameters for causal tree            
    """  
    def __init__(self, X, t, y, cv, params = {}):
        super().__init__(X, t, y, cv)
        self.meta_learner = CausalTreeRegressor(**params)

class CausalForestWrapper(BaseCATEEstimatorWrapper):
    """
    Class for a wrapper around the Causal Forest estimator.
    
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
       Matrix of observed covariates for entire training-validation set.
    t: array-like of shape (n_samples,)
       Vector of treatment assignments for entire training-validation set.
    y: array-like of shape (n_samples,)
       Vector of observed responses for entire training-validation set.
    cv: A StratifiedKFold object used to split the data.
    params: dict
            Dictionary of hyperparameters for causal tree            
    """  
    def __init__(self, X, t, y, cv, params = {}):
        super().__init__(X, t, y, cv)
        self.meta_learner = CausalForest(**params)
        
class EnsembleCATEEstimatorResults(CATEEstimatorResults):
    """
    A container for the results for an ensemble CATE estimator when 
    trained on subset of the training-validation set.
    
    Parameters
    ----------
    list_of_CATE_estimator_results: list
        A list of CATEEstimatorResults objects. All objects are asssumed 
        to have the same train and validation indices.
    """
    def __init__(self, list_of_CATE_estimator_results):
        
        self.train_indices = list_of_CATE_estimator_results[0].train_indices
        self.val_indices = list_of_CATE_estimator_results[0].val_indices
        self._predict_tau(list_of_CATE_estimator_results)

    def _predict_tau(self, list_of_CATE_estimator_results):
        """
        Helper function to predict individual treatment effects for all individuals
        in training-validation set.
        """        
        self.tau = np.mean(np.array([result.tau for result in 
                                     list_of_CATE_estimator_results]), 
                           axis = 0)

class EnsembleCATEEstimatorWrapper(BaseCATEEstimatorWrapper):
    """
    Class for a wrapper around an ensemble CATE estimator.
    
    Parameters
    ----------
    list_of_CATE_estimators: list
        A list of fitted BaseCATEEstimatorWrapper objects with which to form an 
        ensemble. All objects are asssumed to have the same X, t, y, cv inputs.
    """      
    def __init__(self, list_of_CATE_estimators):
        
        self.list_of_CATE_estimators = list_of_CATE_estimators
        self.n_splits = list_of_CATE_estimators[0].n_splits
        self.results = {}
        self.X = list_of_CATE_estimators[0].X
        self.t = list_of_CATE_estimators[0].t
        self.y = list_of_CATE_estimators[0].y
        
    def fit(self):
        """
        Using each split from K fold CV, average the 
        """
        for idx in range(self.n_splits):
            self.results[idx] = \
                EnsembleCATEEstimatorResults([estimator.results[idx]
                                              for estimator in 
                                              self.list_of_CATE_estimators])
            
    def __repr__(self):
        print_string = "Wrapper for an ensemble CATE estimator. Estimators are:"
        for estimator in self.list_of_CATE_estimators:
            print_string += "\n" + str(estimator.meta_learner)
        return print_string
    
def combine_estimator_versions(list_of_versions):
    """
    Combine several fitted wrapper objects for the same estimator. This is to aid
    code simplicity in downstream analysis.
    
    Parameters
    ----------
    list_of_versions: list of CATEEstimatorWrapper objects
        The versions of the same estimator to be combined. We assume these are 
        derived by fitting the estimator using different CV splits.
    
    Returns
    -------
    combined_estimator_wrapper: CATEEstimatorWrapper object
        A single CATEEstimatorWrapper object whose results attribute is the 
        concatenation of those CATEEstimatorWrapper objects comprising the input 
        list_of_versions.
    """
    
    combined_estimator_wrapper = copy.deepcopy(list_of_versions[0])
    n_folds_tot = combined_estimator_wrapper.n_splits
    for additional_version in list_of_versions[1:]:
        for fold, result in additional_version.results.items():
            combined_estimator_wrapper.results[n_folds_tot + fold] = result
        n_folds_tot += additional_version.n_splits
    combined_estimator_wrapper.n_splits = n_folds_tot
        
    return combined_estimator_wrapper