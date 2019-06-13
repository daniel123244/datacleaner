import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import seaborn as sns
from math import pow
from fancyimpute import SimpleFill, KNN, SoftImpute, IterativeSVD, IterativeImputer, MatrixFactorization
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from impyute.imputation.cs import em



############## VISUALIZATION METHODS ###############

class visualization:
    def scatter_matrix(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None,\
                kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None, size=None):
        """ Wrapper for Seaborn pairplot.  Baically never used, just for completion"""
        sns.pairplot(data, hue=hue, hue_order=hue_order, palette=palette, vars=vars, x_vars=x_vars, y_vars=y_vars,\
                kind=kind, diag_kind=diag_kind, markers=markers, height=height, aspect=aspect, dropna=dropna, plot_kws=plot_kws, diag_kws=diag_kws, grid_kws=grid_kws, size=size)
        plt.show()

    def box_plots(data, numerics, nonnumerics):
        """ Takes a list of numeric column names and non-numeric column names.
            returns matrix of box plots (rows = numerics)"""
        if len(numerics) + len(nonnumerics) > 2:
            fig, axes = plt.subplots(nrows=len(numerics), ncols=len(nonnumerics))

            for i in range(len(numerics)):
                for j in range(len(nonnumerics)):
                    sns.boxplot(data[nonnumerics[j]], data[numerics[i]], ax=axes[i][j])
            plt.show()

        else:
            sns.boxplot(data[nonnumerics[0]], data[numerics[0]])
            plt.show()    

    def dist_plots(data, bins = 10, kde = False):
        """ Get all density plots for columns. If kde enabled fit a curve to the plots"""
        _data = data.select_dtypes(include=[np.number])
        ncol = np.floor(np.sqrt(_data.shape[1]))
        nrow = np.ceil(np.sqrt(_data.shape[1]))

        if not _data.shape[1] <= ncol * nrow:
            nrow +=1

        axes = [plt.subplot(ncol, nrow, i+1) for i in range(_data.shape[1])]

        for i in range(_data.shape[1]):
            sns.distplot(_data[_data.columns[i]], bins=bins, kde=kde, ax=axes[i])

        plt.show()


    def correlation_heatmap(data, annot=True):
        """ Produce a correlation heatmap of the given columns"""
        corrs = data.corr()
        sns.heatmap(corrs, annot=annot)
        plt.show()


    def get_correlated_variables(data, dep, sensitivity = 0.3, maxvars = 10):
        """ Get the first max most correlated variables with abs(corelation) over sensitivity"""
        corr_mat = data.corr()
        small_data = np.abs(corr_mat).nlargest(maxvars +1, dep)
        small_data = small_data[small_data[dep] > sensitivity]

        return small_data.index


############## CLEANING AND IMPUTATION METHODS ###############


class cleanimp:
    ############### INITITAL CLEANING
    def describe_missing(data):
        """ Prints number of missing cases and percentage of missing cases by variable."""
        total = data.isnull().sum()
        percent = data.isnull().sum()/data.count()
        missing = pd.concat([data.count(), total, np.round(percent, 3)], axis=1, keys=['Count', 'Total Missing', 'Percent Missing'])
        missing = missing.sort_values(by='Total Missing', ascending=False)
        print(missing)

        case_percent = pd.DataFrame(data=data.isnull().sum(axis=1)/data.count(axis=1), columns = ['Percent Missing'])
        add_perc = pd.concat([data, case_percent], axis=1).sort_values(by = 'Percent Missing', ascending=False)

        print('\n',add_perc.loc[add_perc['Percent Missing']>0])


    def print_missing_cases(data, mask=False):
        """ Prints info for missing cases.  If mask then prints 1 in missing variable spot."""
        mat = data[data.isnull().any(axis=1)] if not mask else data[data.isnull().any(axis=1)].isnull()*1
        print(mat)

    def variable_deletion_analysis(data, max_del=3):
        """ Prints a table of how many complete cases will be left after variable deletion.
            max_del is the maximum of variables to be considered for deletion."""
        deletion_table = pd.DataFrame(data = data.dropna().count().min(), index = ['None'], columns = ['Complete Cases'])

        for i in range(1, max_del + 1):
            combs = combinations(data.columns, i)
            index = [comb for comb in combs]
            comb_table = pd.DataFrame(index = index, columns = ['Complete Cases'])
            for comb in index:
                unpack_comb = [str(col) for col in comb]
                comb_table.loc[comb, 'Complete Cases'] = data.drop(columns=unpack_comb).dropna().count().min()

            deletion_table = pd.concat([deletion_table, comb_table], axis=0)

        #TODO: don't return higher variable counts if lower ones are good enough
        print(deletion_table.loc[deletion_table['Complete Cases'] != deletion_table.loc['None', 'Complete Cases']]\
                                .sort_values(by = 'Complete Cases', axis=0, ascending=False))

    def remove_cases(data, cases):
        data.drop(cases, inplace=True)

    def remove_variable(data, col_name):
        data.drop(col_name, inplace=True)

    ################ TESTING AND IMPUTATION

    def missing_t_tests(data):
        """ Computes a matrix of t-test pvalues for MAR vs MCAR testing.
            Splits each variable into two groups (missing, nonmissing) and
            tests for equal means on all other variables."""

        _data = data.select_dtypes(include=[np.number]) 
        mcar_matrix = pd.DataFrame(data=np.zeros(shape=(_data.shape[1], _data.shape[1])),
                                   columns=_data.columns, index=_data.columns)

        # fill matrix with t-test p values
        for var in _data.columns:
            if _data[var].isnull().any():
                for ovar in _data.columns:
                    part_one = _data.loc[_data[var].isnull(), ovar].dropna()
                    part_two = _data.loc[~_data[var].isnull(), ovar].dropna()
                    mcar_matrix.loc[var, ovar] = stat.ttest_ind(part_one, part_two, equal_var=False).pvalue
            else: #no empty for that var
                mcar_matrix.drop(var, axis=0, inplace=True)

        print(mcar_matrix)

    def little_mcar_test(data):
        """ Compute p-value for Little's MCAR test, taken from pymice.
            Tests if data in general is MCAR"""
        dataset = data.copy()
        dataset = dataset.select_dtypes(include=[np.number])
        vars = dataset.columns
        n_var = dataset.shape[1]

        # mean and covariance estimates
        # ideally, this is done with a maximum likelihood estimator
        gmean = dataset.mean()
        gcov = dataset.cov()

        # set up missing data patterns
        r = 1 * dataset.isnull()
        mdp = np.dot(r, list(map(lambda x: pow(2, x), range(n_var))))
        sorted_mdp = sorted(np.unique(mdp))
        n_pat = len(sorted_mdp)
        correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
        dataset['mdp'] = pd.Series(correct_mdp, index=dataset.index)

        # calculate statistic and df
        pj = 0
        d2 = 0
        for i in range(n_pat):
            dataset_temp = dataset.loc[dataset['mdp'] == i, vars]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
            select_vars = vars[select_vars]
            means = dataset_temp[select_vars].mean() - gmean[select_vars]
            select_cov = gcov.loc[select_vars, select_vars]
            mj = len(dataset_temp)
            parta = np.dot(means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1])))
            d2 += mj * (np.dot(parta, means))

        df = pj - n_var

        # perform test and save output
        p_value = 1 - stat.chi2.cdf(d2, df)

        print('Little\'s MCAR Test p_value: ', p_value)

    def compare_mcar_methods(data, methods = [SimpleFill, KNN, SoftImpute, IterativeSVD, IterativeImputer, MatrixFactorization, em]):
        """ Compare the effects of missing data imputation methods from FancyImpute"""
        _data = data.select_dtypes(include=[np.number])
        index = [method.__name__ for method in methods]
        mean_comp_table = pd.DataFrame(columns = _data.columns, index = index)
        sd_comp_table = pd.DataFrame(columns = _data.columns, index = index)

        for method in methods:
                if method.__name__ != 'SimpleFill' and method.__name__ != 'em':
                    mean_comp_table.loc[method.__name__] = pd.DataFrame(method(verbose=False).fit_transform(_data), columns=_data.columns, index=_data.index).mean()
                    sd_comp_table.loc[method.__name__] = pd.DataFrame(method(verbose=False).fit_transform(_data), columns=_data.columns, index=_data.index).std()
                elif method.__name__ == 'SimpleFill':
                    mean_comp_table.loc[method.__name__] = pd.DataFrame(method().fit_transform(_data), columns=_data.columns, index=_data.index).mean()
                    sd_comp_table.loc[method.__name__] = pd.DataFrame(method().fit_transform(_data), columns=_data.columns, index=_data.index).std()

                elif method.__name__ == 'em':
                    e = em(_data)
                    e.index = _data.index
                    e.columns = _data.columns
                    e_mean = e.mean()
                    e_std = e.std()
                    mean_comp_table.loc[method.__name__] = e_mean
                    sd_comp_table.loc[method.__name__] = e_std

        # include no transofrmation and drop columns with no nulls
        # this has to go here because some fancyimpute methods rely
        # on other columns
        mean_comp_table = mean_comp_table.append(pd.DataFrame(_data.mean(), columns=['None']).T)
        null_cols = pd.DataFrame(_data.isnull().any(axis=0), columns=['y'])
        null_cols = null_cols.loc[~null_cols.y].index
        mean_comp_table.drop(columns = null_cols, inplace= True)
        sd_comp_table = sd_comp_table.append(pd.DataFrame(_data.std(), columns=['None']).T)
        sd_comp_table.drop(columns = null_cols, inplace= True)

        print('Estimates of Means')
        print(mean_comp_table)

        print('\nEstimates of Standard Deviation')
        print(sd_comp_table)


    def impute_values(data, method = SimpleFill):
        # fit and transform the selected columns using selected imputation method
        if not type(cols) == list:
            raise ValueError('Cols needs to be a list.')

        if method.__name__ != 'SimpleFill' and method.__name__ != 'em':
            data = pd.DataFrame(method(verbose=False).fit_transform(data), columns=data.columns, index=data.index)

        elif method.__name__ == 'SimpleFill':
            data = pd.DataFrame(method().fit_transform(data), columns=data.columns, index=data.index)

        elif method.__name__ == 'em':
            e =  em(data)
            e.index = data.index
            e.columns= data.columns
            data = e


############## OUTLIER DETECTION METHODS ###############

class outlier:
    def univariate_outlier_analysis(data, sd = 3):
        """ Finds univariate outliers by scaling and looking at t-scores"""

        # maybe this should tell you which outliers go with which column..
        _data = data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled = pd.DataFrame(np.abs(scaler.fit_transform(_data)), index=_data.index)

        print(_data.loc[(scaled > sd).any(axis=1)])
        

    def __ellipse_data(dep, indep, sd):
        """ Helper function to draw an ellipse on the principal components
            of a bivariate data plot"""
        centroid = (dep.mean(), indep.mean())
        cov = pd.concat([dep, indep], axis=1).cov()

        # get eigenvectors and values and sort them largest to smallest
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # get the angle of rotation for the ellipse
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        w,h = 2*sd*np.sqrt(vals)

        return Ellipse(xy=centroid, width=w, height=h, angle=theta, color='black')


    def outlier_scatterplot(data, dep, indep, sd = 3):
        """ Returns a scatterplot with principal component ellipse super imposed."""
        ax = sns.scatterplot(data[dep], data[indep])

        el = outlier.__ellipse_data(data[dep], data[indep], sd)
        el.set_facecolor('none')
        ax.add_artist(el)

        plt.show()


    def bivariate_outlier_analysis(data, sd = 3):
        """ Returns a scattermatrix of data[cols] with principal component ellipses superimposed.
            Used to find specific plots to investigate with bivariate_outlier_ellipse"""
        axes = sns.pairplot(data).axes
        
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i != j:
                    el = outlier.__ellipse_data(data[data.columns[j]], data[data.columns[i]], sd)
                    el.set_facecolor('none')
                    axes[i][j].add_artist(el)

        plt.show()


    def multivariate_outlier_analysis(data, sd = 2.5):
        """ This is an implemenation of the Mahalanobis D^2 test.
            We compute the Mahalanobis distance for each observation and divide by the
            number of variables in our dataset.  Hair et all 2013 says this is distributed
            roughly as a t-value, so we look for t valyes greated than sd.
            WARNING: at this point we expect NO missing data"""
        
        # center the data
        _data = data.select_dtypes(include=[np.number])
        cdata = _data - _data.mean()
        cov = cdata.cov()
        invcov = np.linalg.inv(cov)
        D2 = np.diag(np.dot(cdata.dot(invcov), cdata.T))
        D2overDF = D2/_data.shape[1]

        MD = pd.DataFrame(D2overDF, index=data.index, columns=['D2_Scores'])

        print(MD[MD.D2_Scores > sd])



############## ASSUMPTION TESTING METHODS ###############
class assumptions:
    def residual_scatter_plot(data, dep, indep):
        """ Wrapper for seaborn residualplot"""
        sns.residplot(data[dep], data[indep], lowess=True)
        plt.show()

    def normal_plots(data):
        """ Plot a matrix of normal probability plots for normaility testing """
        _data = data.select_dtypes(include=[np.number]).dropna()
        length = _data.shape[1]
        ncol = np.floor(np.sqrt(length))
        nrow = np.ceil(np.sqrt(length))

        if not length <= ncol * nrow:
            nrow +=1

        axes = [plt.subplot(ncol, nrow, i+1) for i in range(length)]

        for i in range(length):
            stat.probplot(_data[_data.columns[i]].loc[_data[_data.columns[i]].notnull()], plot=axes[i], fit=True)
            axes[i].set_title(_data.columns[i])

        plt.show()


    def normal_stat_tests(data):
        """ Get the z-scores for skew and kurtosis.
            WARNING: NO MISSING DATA"""

        _data = data.select_dtypes(include=[np.number]).dropna()
        skew_factor = np.sqrt(6/_data.count())
        skew_series = _data.skew()/skew_factor

        kurt_factor = np.sqrt(24/_data.count())
        kurt_series = _data.kurt()/kurt_factor

        normal_pvals = pd.Series(stat.normaltest(_data).pvalue, index = _data.columns)

        pvals = pd.concat([skew_series, kurt_series, normal_pvals], axis=1)
        pvals.columns = ['Skew z-value', 'Kurtosis z-value', 'Normality p-value']
        print(pvals)


    def test_multicolinearity(data, cor_vars):
        """ Overall test of multicolinearity using VIF from ISLR"""
        raise NotImplementedError('Oops, I\'ll get to this later. Maybe...')