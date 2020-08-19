import numpy as np

def get_Neyman_ATE(y, t):
    """
    Compute the Neyman ATE estimate w.r.t. a binary treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    ATE_: float
       The Neyman ATE estimate.
    """
    y0_obs = y[t==0]
    y1_obs = y[t==1]
    ATE_ = y1_obs.mean() - y0_obs.mean()
    return ATE_


def get_Neyman_var(y, t):
    """
    Compute the Neyman ATE variance estimate w.r.t. a binary treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    V_Neyman: float
       The Neyman ATE sampling variance estimate.
    """
    N_c = (1-t).sum()
    N_t = t.sum()
    y0_obs = y[t==0]
    y1_obs = y[t==1]
    
    var_c_ = np.var(y0_obs, ddof=1)
    var_t_ = np.var(y1_obs, ddof=1)
    V_Neyman = var_c_/N_c + var_t_/N_t
    
    return V_Neyman

def get_subgroup_CATE(y, t, subgroup_indicator):
    """
    Compute the Neyman estimate for the CATE for a subgroup (treatment effect 
    averaged over the subgroup.)
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
        Vector of observed responses
    t: array-like of shape (n_samples,)
        Vector of treatment assignments taking values in {0,1}
    subgroup_indicator: array-like of shape (n_samples,)
        Indicator vector for a subgroup
    
    Returns
    -------
    CATE_: float
           The CATE for the subgroup defined by the indicator.
    """
    
    # Compute y, t vectors for the subgroup
    y_sg = y[subgroup_indicator]
    t_sg = t[subgroup_indicator]
    CATE_ = get_Neyman_ATE(y_sg, t_sg)
    
    return CATE_

def get_subgroup_CATE_std(y, t, subgroup_indicator):
    """
    Compute the std estimate for the Neyman CATE estimator for a subgroup 
    (treatment effect averaged over the subgroup.)
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
        Vector of observed responses
    t: array-like of shape (n_samples,)
        Vector of treatment assignments taking values in {0,1}
    subgroup_indicator: array-like of shape (n_samples,)
        Indicator vector for a subgroup
    
    Returns
    -------
    CATE_: float
           The CATE for the subgroup defined by the indicator.
    """
    
    # Compute y, t vectors for the subgroup
    y_sg = y[subgroup_indicator]
    t_sg = t[subgroup_indicator]
    CATE_std_ = np.sqrt(get_Neyman_var(y_sg, t_sg))
    
    return CATE_std_


def get_subgroup_t_statistic(y, t, subgroup_indicator, 
                             dataset_indicator = None):
    """
    Compute the t-statistic for a subgroup. t-statistic to be defined in paper.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
        Vector of observed responses
    t: array-like of shape (n_samples,)
        Vector of treatment assignments taking values in {0,1}
    subgroup_indicator: array-like of shape (n_samples,)
        Indicator vector for a subgroup
    datasset_indicator: array-like of shape (n_samples,)
        Indicator vector for the validation set / training set if we would
        like to compute the t-statistic w.r.t these sets
    
    Returns
    -------
    t_statistic: float
           The t_statistic for the subgroup defined by the indicator.
    """
    
    # Restrict the dataset if so desired
    if dataset_indicator is not None:
        assert len(dataset_indicator) == len(y)
        y = y[dataset_indicator]
        t = t[dataset_indicator]
        subgroup_indicator = subgroup_indicator[dataset_indicator]
    else:
        pass
    
    # Compute Neyman estimates for the ATE and subgroup CATE
    ATE_ = get_Neyman_ATE(y, t)
    CATE_ = get_subgroup_CATE(y, t, subgroup_indicator)
    
    # Compute y, t vectors for both the subgroup and its complement.
    y_sg = y[subgroup_indicator]
    t_sg = t[subgroup_indicator]
    y_rest = y[~subgroup_indicator]
    t_rest = t[~subgroup_indicator]
    
    # Compute y_obs vectors and their lengths for both the subgroup and
    # its complement
    y0_obs_sg = y_sg[t_sg==0]
    y1_obs_sg = y_sg[t_sg==1]
    y0_obs_rest = y_rest[t_rest==0]
    y1_obs_rest = y_rest[t_rest==1]
    N_c_sg = len(y0_obs_sg)
    N_t_sg = len(y1_obs_sg)
    N_c_rest = len(y0_obs_rest)
    N_t_rest = len(y1_obs_rest)
    N_c = N_c_sg + N_c_rest
    N_t = N_t_sg + N_t_rest
    
    # Try computing the variance estimate, if it fails (because sample size is 
    # too small), return nan
    try:
        var_sg_ = np.var(y0_obs_sg, ddof = 1) * N_c_sg * (1/N_c_sg - 1/N_c)**2 \
            + np.var(y1_obs_sg, ddof = 1) * N_t_sg * (1/N_t_sg - 1/N_t)**2
        var_rest_ = np.var(y0_obs_rest, ddof = 1) * N_c_rest / N_c**2 \
            + np.var(y1_obs_rest, ddof = 1) * N_t_rest / N_t**2
    except:
        return(np.NaN)
    
    t_statistic = (CATE_ - ATE_) / np.sqrt(var_sg_ + var_rest_)
    
    return t_statistic


def get_relative_risk(y, t):
    """
    Compute the plug-in estimate for relative risk of a treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    RR_: float
       The plug-in estimate for relative risk.
    """
    Y0_obs = y[t==0]
    Y1_obs = y[t==1]
    RR_ = Y1_obs.mean() / Y0_obs.mean()
    
    return RR_

def get_relative_risk_CI(y, t):
    """
    Compute a 95% CI for the plug-in estimate for relative risk of a treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    RR_CI: array-like of shape (2,)
       A 95% CI for the plug-in estimate for relative risk.
    """
    def get_log_RR_var(y, t):
        """
        Helper function to compute the variance for log(plug-in estimate for RR) 
        using the delta method.
        """
        Y0_obs = y[t==0]
        Y1_obs = y[t==1]
        n0 = len(Y0_obs)
        n1 = len(Y1_obs)
        x0 = Y0_obs.sum()
        x1 = Y1_obs.sum()
        
        return 1/x0 - 1/n0 + 1/x1 - 1/n1
    
    RR_ = get_relative_risk(y, t)
    log_RR_var = get_log_RR_var(y, t)
    log_RR_CI = np.array((np.log(RR_) - 1.96 * np.sqrt(log_RR_var),
                          np.log(RR_) + 1.96 * np.sqrt(log_RR_var)))
    RR_CI = np.exp(log_RR_CI)
    
    return RR_CI