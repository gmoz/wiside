import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from modeling import logger_info, logger_error
from opendata import Temper_Precp

# --- Use Sample data to replace ---- #
# def concatenate_data(true_data, res_wifi_stay_df, res_blue_stay_df, vendors_wifi, vendors_blue, res_wifi_df, res_blue_df):
#     """Concatenate the true_data(Y), all explanatory variables(X) which contains open data and Wi-Fi & Bluetooth data."""
#     join_tbl = res_wifi_stay_df.merge(true_data, left_index= True, right_index= True) \
#                                .merge(Temper_Precp, left_index= True, right_index= True) \
#                                .merge(res_blue_stay_df, left_index= True, right_index= True) \
#                                .merge(pd.concat({ 'vendors_wifi': vendors_wifi, 'vendors_blue': vendors_blue}, axis =1),
#                                  left_index= True, right_index= True) \
#                                .merge(pd.concat({ 'signal_wifi': res_wifi_df['number_of_det'], 'signal_blue': res_blue_df['number_of_det']}, axis =1),
#                                  left_index= True, right_index= True)
#     join_tbl['weekday'] = join_tbl.index.weekday
#     # non business day, e.g. Monday 
#     count_cond = (join_tbl['weekday'] != 0) 
#     join_tbl = join_tbl[count_cond]
#     # event
#     join_tbl['event'] = (~join_tbl['備註'].isna()).apply(lambda x: int(x))
#     # weekday
#     import category_encoders as ce
#     encoder_weekday = ce.OneHotEncoder(cols = ['weekday']).fit(join_tbl)
#     new_join_tbl = encoder_weekday.transform(join_tbl)
#     # split response & explanatory ,drop unused columns
#     response_variable = new_join_tbl.loc[:,'當日入園人數合計']
#     explanatory_var = [ '當日入園人數合計', 'wifi_person', 'blue_person','vendors_wifi', 'vendors_blue', 'signal_wifi', 'signal_blue','Temperature', 'Precp']
#     explanatory_var.extend(['event'])
#     explanatory_var.extend(['weekday_' + str(i) for i in range(1,7) ])
#     new_join_tbl = new_join_tbl.loc[:, explanatory_var ]
#     new_join_tbl = new_join_tbl.drop(response_variable.name, axis = 1)
#     print('After concate_data, (n,p) = ({}, {}).'.format(*new_join_tbl.shape))
#     logger_info.info('After concate_data, (n,p) = ({}, {}).'.format(*new_join_tbl.shape))
#     return new_join_tbl, response_variable
# X, y = concatenate_data(true_data, res_wifi_stay_df, res_blue_stay_df, vendors_wifi, vendors_blue, res_wifi_df, res_blue_df))
df = pd.read_csv(r"./data_cache/Sample_data.csv", sep = ',', index_col= 0, parse_dates = True)
X, y = df.iloc[:,1:], df.iloc[:,0]
# --- Use Sample data to replace ---- #

## Split into Train & Test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0, shuffle = True, stratify = y.index.weekday)

class Polynomial_feature(BaseEstimator, TransformerMixin):
    
    def __init__(self, degree, interaction_only = True, number_feature = None):
        self.degree = degree
        self.interaction_only = interaction_only
        self.number_feature = number_feature
        
    def fit(self, X, y = None):
        self.named_feature = X.columns
        return self
    
    def transform(self, X):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree = self.degree, interaction_only=True, include_bias= False)
        poly_feature = poly.fit_transform(X.iloc[:, 0:self.number_feature ])
        self.poly_feature_name = poly.get_feature_names( self.named_feature[0:self.number_feature] )
        return np.concatenate((
                        poly_feature,
                        X.iloc[:, self.number_feature: ]),
                        axis = 1)

def build_pipeline(model = None, Scaler = None, verbose = 1):
    '''To Adapt different model.'''
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ('Polynomial_feature', Polynomial_feature(degree= 2, interaction_only= True, number_feature= 2)),
        ('Scaler', Scaler),
        ('model', model) ], verbose = verbose)

def build_grid_search(pipeline, param_grid, k_fold = 2, n_jobs = None):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, make_scorer
    return GridSearchCV(pipeline, param_grid, cv = k_fold,
                        return_train_score = True,
                        refit = 'MAE',
                        scoring = {
                            'MSE': make_scorer(mean_squared_error, greater_is_better= False), # -MSE
                            'MAE': make_scorer(mean_absolute_error, greater_is_better= False), # -MAE
                        },
                        verbose = 1, n_jobs = n_jobs)

def run_grid_search(grid_search, sample_weight = None):
    if sample_weight is None:
        grid_search.fit(X_train, y_train)
    else:
        grid_search.fit(X_train, y_train, **{'model__sample_weight' : sample_weight})

    print('Best test score accuracy is:', grid_search.best_score_)
    logger_info.info('Best test score accuracy is: {}'.format(grid_search.best_score_))
    return pretty_cv_results(grid_search.cv_results_)

def pretty_cv_results(cv_results, sort_by ='rank_test_MAE', sort_ascending =True, n_rows = 10):
    df = pd.DataFrame(cv_results)
    cols_of_interest = [key for key in df.columns if key.startswith('param_') 
                        or key.startswith('mean_train') 
                        or key.startswith('mean_test_')
                        or key.startswith('rank')]
    return df.loc[:, cols_of_interest].sort_values(by=sort_by, ascending=sort_ascending).head(n_rows)
    
def test_performance(gridRes_or_estimator, input_est = 'grid_search_est', Plot_name = None):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, make_scorer
    def err(y_pred, y_true):
        """MAE_ration: custom error function. **[ abs(y_pred - y_true) / y_true ]** """
        Index = y_true.index     
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy()
        
        ABS = np.abs(y_pred - y_true)
        ABS_weight = np.abs(y_pred - y_true) / np.array(y_true)
        
        print("Date, abs(y_pred - y_true) and [ abs(y_pred - y_true) / y_true ].")
        logger_info.info("Date, abs(y_pred - y_true) and [ abs(y_pred - y_true) / y_true ].")
        for i, j, k, l in zip(Index, ABS, ABS_weight, y_true):
            print(f"{i}, ({l}, {round(j,0)} and {k*100:.1f}%).")
            logger_info.info(f"{i}, ({l}, {round(j,0)} and {k*100:.1f}%).")
        
        return sum( np.abs(y_pred - y_true) / np.array(y_true) ) / len(y_pred) *100
    
    if input_est == 'grid_search_est':
        print(f"## Final estimator select by CV is: \n {gridRes_or_estimator.best_estimator_.get_params()}")
        logger_info.info(f"## Final estimator select by CV is: \n {gridRes_or_estimator.best_estimator_.get_params()}")
        best_est = gridRes_or_estimator.best_estimator_.fit(X_train, y_train)
    elif input_est == 'estimator':
        best_est = gridRes_or_estimator.fit(X_train, y_train)
    
    print("## Training error is (RMSE, MAE, MAE_Ytrue) = (%.3f, %.3f, %.3f)" % (
        mean_squared_error(best_est.predict(X_train), y_train, squared= False),
        mean_absolute_error(best_est.predict(X_train), y_train),
        err(best_est.predict(X_train), y_train)))
    logger_info.info("## Training error is (RMSE, MAE, MAE_Ytrue) = (%.3f, %.3f, %.3f)" % (
        mean_squared_error(best_est.predict(X_train), y_train, squared= False),
        mean_absolute_error(best_est.predict(X_train), y_train),
        err(best_est.predict(X_train), y_train)))
    
    print("## Testing error is (RMSE, MAE, MAE_Ytrue) = (%.3f, %.3f, %.3f)" % (
        mean_squared_error(best_est.predict(X_test), y_test, squared= False),
        mean_absolute_error(best_est.predict(X_test), y_test),
        err(best_est.predict(X_test), y_test)))
    logger_info.info("## Testing error is (RMSE, MAE, MAE_Ytrue) = (%.3f, %.3f, %.3f)" % (
        mean_squared_error(best_est.predict(X_test), y_test, squared= False),
        mean_absolute_error(best_est.predict(X_test), y_test),
        err(best_est.predict(X_test), y_test)))
        
    ###############
    # Plot result
    plt.plot([x for x in range(2*10**3)], [x for x in range(2*10**3)])
    plt.scatter(best_est.predict(X_train), y_train)
    plt.plot([x for x in range(2*10**3)], [x for x in range(2*10**3)])
    plt.scatter(best_est.predict(X_test), y_test)
    plt.show()

    import plotly
    from plotly.graph_objs import Figure, Scatter, Layout
    fig = Figure()
    fig.add_trace(Scatter(x = best_est.predict(X_train), y = y_train, text = list(y_train.index), mode= 'markers',
                          name = 'train'))
    fig.add_trace(Scatter(x = best_est.predict(X_test), y = y_test, text = list(y_test.index), mode= 'markers',
                          name = 'test'))
    fig.add_trace(Scatter(x = [0, 2*10**3], y = [0, 2*10**3], name = 'Y = X', mode= 'lines'))
    fig.update_traces(marker_line_width=2, marker_size=10) 
    fig.update_layout(colorway = ["#46A3FF", "#53FF53", "#FF5151"],
                  title= Plot_name,
                  yaxis_zeroline=False, xaxis_zeroline=False,
                  yaxis_tickmode = 'auto', yaxis_nticks = 20, yaxis_type = 'linear', yaxis_ticksuffix = "人次",
                  yaxis_exponentformat =  "none", yaxis_rangemode = "normal",
                  yaxis_title = "實際入場人次", 
                  xaxis_tickmode = 'auto', xaxis_nticks = 20, xaxis_type = 'linear', xaxis_ticksuffix = "人次",
                  xaxis_exponentformat =  "none", xaxis_rangemode = "normal",
                  xaxis_title = "推測入場人次",
                  xaxis_tickangle = 45)
    fig.show()