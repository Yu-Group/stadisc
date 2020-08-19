import pandas as pd

def prepare_df(path, features, response_var, treatment_var = "TREATED"):
    """
    Load the data frame with the specified features, treatment variable, and response variable
            
    Parameters
    ----------
    path : str; path to the raw data file
    features: list of str; list of features to keep
    response_var: str; column to use as the response
    treatment_var: str; column to use as treatment variable
    
    Returns
    -------
    data_df: array-like; cleaned data frame
    """
    
    data_df = pd.read_csv(path)
    data_df = data_df[features + [treatment_var] + [response_var]]
    
    return data_df

def separate_vars(input_df, response_var, treatment_var = "TREATED"):
    """
    Separate the data frame into a treatment vector, a repsonse vector, and a remaining data frame of other covariates
    
    Parameters
    ----------
    input_df : str; the input data frame
    response_var: str; column to use as the response
    treatment_var: str; column to use as treatment variable
    
    Returns
    -------
    X: array-like
    t: array-like
    y: array-like
    """    

    t = input_df[treatment_var].copy().values
    y = input_df[response_var].copy().values
    X = input_df.drop(columns = [treatment_var, response_var]).copy().values
    
    return X, t, y

