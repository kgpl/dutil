import warnings
import numpy as np
import pandas as pd

def __filter__(dataframe, filter):
    df = dataframe.copy()
    if filter is not None:
        try:
            df = dataframe[dataframe.eval(filter)].copy()
        except:
            warnings.warn("Unable to apply filter.")
    return df

def __desc__(df, conf=None):
    summ = df.describe(include='all').transpose()[['count', 'mean', 'std', 'min', 'max']]
    summ['median'] = df.median()
    summ['unique'] = df.nunique()
    summ = summ[['count', 'unique', 'min', 'max', 'mean', 'std', 'median']]
    z = {'80%':1.282, '85%':1.440, '90%':1.645, '95%':1.960, '99%':2.576, '99.5%':2.807, '99.9%':3.291}
    if conf in z.keys():
        summ['%s Confidence Interval Low'%(conf)] = summ['mean'] - ((z[conf] * summ['std'])/(summ['count']**(1/2)))
        summ['%s Confidence Interval High'%(conf)] = summ['mean'] + ((z[conf] * summ['std'])/(summ['count']**(1/2)))
    else:
        warnings.warn("required confidence interval not in supported list.\nUse one of: {}".format(z.keys()))
    return summ

def summary(dataframe, column=None, filter=None, group=None, conf=None):
    # Filter
    df = __filter__(dataframe, filter)
    # Columns
    if column is not None:
        # Verify if column and group is correct type
        if isinstance(column, str):
            columns = [column]
        elif isinstance(column, list):
            columns = column
        else:
            raise TypeError("column should be of either list or string.")

        if not all(x in df.columns for x in columns):
            raise ValueError("All mentioned columns don't exist in dataframe.")

        if group is None:
            groups = []
        elif isinstance(group, str):
            groups = [group]
        elif isinstance(group, list):
            groups = group
        else:
            raise TypeError("group should be of either list or string.")
        
        if not all(x in df.columns for x in groups):
            raise ValueError("All mentioned groups don't exist in dataframe.")

        # sorting dataframe
        df = df[columns + groups]
        if len(groups)==0:
            summ = __desc__(df, conf)
        else:
            summ = pd.DataFrame()
            grps = df.groupby(groups)
            for nn, gg in grps:
                temp = pd.DataFrame()
                temp = temp.append(__desc__(gg[columns], conf))
                if isinstance(nn, str):
                    nn = [nn]
                for itr in range(0, len(nn)):
                    temp.insert(0, groups[itr], nn[itr])
                summ = summ.append(temp)
            summ = summ.drop(['unique'], axis = 1)
            summ.insert(0, 'Property', summ.index)
            summ = summ.sort_values(by=['Property']+groups)
            summ = summ.reset_index(drop=True)
    else:
        summ = __desc__(df, conf)
    
    return summ

def polyfit(dataframe, X, Y, filter=None, deg=1):
    # Filter
    df = __filter__(dataframe, filter)
    if X not in df.columns:
        raise ValueError("Column 'X' don't exist in dataframe.")
    if Y not in df.columns:
        raise ValueError("Column 'Y' don't exist in dataframe.")
    df = df[[X,Y]]
    df = df.sort_values(by=[X])
    x = np.array(df[X])
    y = np.array(df[Y])
    return np.polyfit(x, y, deg)