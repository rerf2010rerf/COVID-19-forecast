import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.stats import linregress
from scipy.misc import derivative
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import namedtuple
from scipy.optimize import minimize, dual_annealing, shgo, differential_evolution, basinhopping
from tqdm.notebook import tqdm

def read_data_for_country(country, province, directory):
    rus_confirmed = {}
    rus_death = {}
    rus_recovered = {}

    for file in os.listdir(directory):
        if file.endswith('csv'):
            temp = pd.read_csv(directory+file)
            if 'Province/State' in temp.columns:
                death_series = temp.loc[(temp['Country/Region']==country) & (temp['Province/State']==province), 'Deaths']
                confirmed_series = temp.loc[(temp['Country/Region']==country) & (temp['Province/State']==province), 'Confirmed']
                recovered_series = temp.loc[(temp['Country/Region']==country) & (temp['Province/State']==province), 'Recovered']
            else:
                death_series = temp.loc[(temp['Country_Region']==country) & (temp['Province_State']==province), 'Deaths']
                confirmed_series = temp.loc[(temp['Country_Region']==country) & (temp['Province_State']==province), 'Confirmed']
                recovered_series = temp.loc[(temp['Country_Region']==country) & (temp['Province_State']==province), 'Recovered']

            time = pd.Timestamp(file[:-4])

            if confirmed_series.shape[0] > 0:
                rus_confirmed[time] = confirmed_series.iloc[0]
            if death_series.shape[0] > 0:
                rus_death[time] = death_series.iloc[0]
            if recovered_series.shape[0] > 0:
                rus_recovered[time] = recovered_series.iloc[0]

    rus_confirmed = pd.Series(rus_confirmed).fillna(0).astype(int)
    rus_death = pd.Series(rus_death).fillna(0).astype(int)
    rus_recovered = pd.Series(rus_recovered).fillna(0).astype(int)
    
    return pd.DataFrame({'confirmed': rus_confirmed.values, 
                         'death': rus_death.values, 
                         'recovered': rus_recovered.values},
                       index=rus_confirmed.index)

def read_timeseries_for_country(country, directory):
    if country in ['China', 'Canada', 'Australia']:
        return read_timeseries_for_country_all_provinces(country, directory)
    
    conf_data = pd.read_csv(directory + 'time_series_covid19_confirmed_global.csv')
    recov_data = pd.read_csv(directory + 'time_series_covid19_recovered_global.csv')
    death_data = pd.read_csv(directory + 'time_series_covid19_deaths_global.csv')

    index = [pd.Timestamp(d) for d in conf_data.columns[4:]]
#     return pd.DataFrame({'confirmed': conf_data[conf_data['Country/Region']==country][4:]})
    conf_data = conf_data[(conf_data['Country/Region']==country) & (conf_data['Province/State'].isna())].iloc[0,4:]
    recov_data = recov_data[(recov_data['Country/Region']==country) & (recov_data['Province/State'].isna())].iloc[0,4:]
    death_data = death_data[(death_data['Country/Region']==country) & (death_data['Province/State'].isna())].iloc[0,4:]
    
    return pd.DataFrame({'confirmed': conf_data.values, 
                        'recovered': recov_data.values,
                        'death': death_data.values},
                       index=index).astype(int)

def read_timeseries_for_country_all_provinces(country, directory):
    conf_data = pd.read_csv(directory + 'time_series_covid19_confirmed_global.csv')
    recov_data = pd.read_csv(directory + 'time_series_covid19_recovered_global.csv')
    death_data = pd.read_csv(directory + 'time_series_covid19_deaths_global.csv')

    index = [pd.Timestamp(d) for d in conf_data.columns[4:]]
#     return pd.DataFrame({'confirmed': conf_data[conf_data['Country/Region']==country][4:]})
    conf_data = conf_data[(conf_data['Country/Region']==country)].iloc[:,4:]
    recov_data = recov_data[(recov_data['Country/Region']==country)].iloc[:,4:]
    death_data = death_data[(death_data['Country/Region']==country)].iloc[:,4:]
    
    return pd.DataFrame({'confirmed': conf_data.sum().values, 
                        'recovered': recov_data.sum().values,
                        'death': death_data.sum().values},
                       index=index).astype(int)
                       
                       
                       
deriv_count = 0
def solve_seir(S0, E0, I0, R0, betas, gamma, alpha, mu, times):
    N0 = S0+E0+I0+R0
    global deriv_count
    deriv_count = 0
    
    
    def deriv(t, y, betas, gamma, alpha, mu):
        global deriv_count
        deriv_count+=1
        S, E, I, R = y
        N = S+E+I+R
        dSdt = - betas[int(t)]*S*I/N
        dEdt = betas[int(t)]*S*I/N - alpha*E
        dIdt = alpha*E - gamma*I
        dRdt = gamma*I
        return dSdt, dEdt, dIdt, dRdt
    
    y0 = S0, E0, I0, R0
    ret = solve_ivp(deriv, t_span=(0, times[-1]), y0=y0, t_eval=times, args=(betas, gamma, alpha, mu,),
                   )
    S, E, I, R = ret.y
    return S, E, I, R


def plot_ir(I, R, country, country_name, t, ax=None):
    N = np.max(I+R)
    rusN = np.max(country)
    N = np.max([rusN, N])
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    if ax is None:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, I+R, 'r', alpha=0.5, lw=2, label='Infectious+Recovered')
    ax.plot(t, I, 'g', alpha=0.5, lw=2, label='Infectious')
    ax.plot(t, R, 'b', alpha=0.5, lw=2, label='Recovered')
    ax.plot(np.arange(0, country.shape[0], 1), country, 'black', alpha=0.5, lw=2, label=country_name)
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.set_ylim(0, N+N*0.1)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    return ax
    
    
def max_metric(df1, df2):
    return np.max(np.abs(df1-df2))

def eucl_metric(df1, df2):
    return np.linalg.norm(df1-df2)

def male_metric(df1, df2):
    errors = ((df1+1)/(df2+1)).apply(np.log10).abs()
    return errors.mean()

# def male_weighted_metric(df1, df2, koef):
#     errors = ((df1+1)/(df2+1)).apply(np.log10).abs()
#     k=errors.shape[0]
#     x = np.arange(-k/2, k/2, 1)
#     y = 1/(1+np.exp(-x/(k*koef)))
    
#     errors = errors * y
#     return errors.mean()


def dFdt_right(func_df):
    return (func_df.shift(-1) - func_df)[:-1]

def dFdt_left(func_df):
    return (func_df - func_df.shift(1))[1:]

def dFdt_twosided(func_df):
    result = np.array([derivative(lambda x: func_df.iloc[x], x0, dx=1, n=1) for x0 in range(1, func_df.shape[0]-1)])
    result = np.hstack([func_df.iloc[1]-func_df.iloc[0], result, func_df.iloc[-1]-func_df.iloc[-2]])
    return pd.Series(result, index=func_df.index)

# def dFdt_twosided(func_df):
#     result = np.array([derivative(lambda x: func_df.iloc[x], x0, dx=1, n=1) for x0 in range(1, func_df.shape[0]-1)])
#     result = np.hstack([result[0], result, result[-1]])
#     return pd.Series(result, index=func_df.index)


def find_gamma(data_df, metric=male_metric):
    dRdt = dFdt_twosided(data_df.recovered) # dR/dt ~ R(t+1)-R(t)
    I = (data_df.confirmed - data_df.recovered)
    min_gamma = 0
    min_err = 10000000000
    for gamma in np.logspace(-2, 0, 500):
        err = metric(dRdt, gamma*I)
        if err < min_err:
            min_err = err
            min_gamma = gamma
    return min_gamma, min_err
    
    
    
    
seir_params = namedtuple('seir_params', ['S', 'E', 'I', 'R', 'N', 'gamma', 'beta', 'gamma_err'])

def compute_scalar_params(data_df, N):
    gamma, gamma_err = find_gamma(data_df)
    I = data_df.confirmed - data_df.recovered
    dIdt = dFdt_twosided(I)
    
    gammaI = gamma*I
    E = (dIdt + gammaI)/alpha
    
    R = data_df.recovered
    S = N-I-E-R # S+E = N-Confirmed
    
    dSdt = dFdt_twosided(S)
    IS_div_N = I*S/N
    beta = -dSdt/IS_div_N # dSdt = -beta*IS_div_N
    
    return seir_params(S, E, I, R, N, gamma, beta, gamma_err)

def cut_confirmed(data_df, cut_koef=10):
    max_conf = data_df.confirmed.max()
    if max_conf < 1000:
        print(f'\t Warning: Not enough Infectious observations for cutting: max_confirmed = {max_conf}')
    for i, val in enumerate(data_df.confirmed):
        if val >= max_conf/cut_koef:
            break
    result = data_df.iloc[i:,:]
    if result.shape[0] < 20:
        print(f'\t Warning: Not enough observations left after cutting: {result.shape[0]}')
    return result

def cut_confirmed_to_number(data_df, number=10):
    max_conf = data_df.confirmed.max()
    if max_conf < 1000:
        print(f'\t Max number of infectious observations: max_confirmed = {max_conf}')
    result = data_df[data_df.confirmed > 0]

    result = data_df.iloc[-number:,:]
    if result.shape[0] < number:
        print(f'\t Warning: Not enough observations left after cutting: {result.shape[0]}')
        new_val = pd.DataFrame({'confirmed': np.zeros(number-result.shape[0]), 
                                          'death': np.zeros(number-result.shape[0]), 
                                          'recovered': np.zeros(number-result.shape[0])})
        new_val.index = [result.index[-1] - pd.Timedelta(f'{n} days') for n in range(1, number-result.shape[0]+1)][::-1]
        result = pd.concat([new_val, result])
    return result
    
    

def get_step_func(timesunits, x1, x2, middle):
    return np.hstack([np.ones(middle)*x1, np.ones(timesunits*100-middle)*x2])

def get_twostep_func(timesunits, x1, x2, x3, middle1, middle2):
    if middle1 > middle2:
        middle1, middle2 = middle2, middle1
    return np.hstack([
        np.ones(middle1)*x1,
        np.ones(middle2-middle1)*x2,
        np.ones(timesunits*100-middle2)*x3
    ])

def onestep_beta_cost_func(x1, x2, middle, data_df, seir_params):
    timesunits = seir_params.S.shape[0]
    times = np.arange(0, seir_params.S.shape[0], 1)
    S, E, I, R = solve_seir(
        S0=seir_params.S.iloc[0], 
        E0=seir_params.E.iloc[0],
        I0=seir_params.I.iloc[0],
        R0=seir_params.R.iloc[0],
                betas = get_step_func(timesunits, x1, x2, int(middle)),
               gamma=seir_params.gamma, alpha=alpha, mu=0, times=times
              )

    return male_metric(I+R, data_df.confirmed)


def get_zerostep_func(timesunits, x1):
    return np.ones(timesunits*100)*x1


def zerostep_beta_cost_func(x1, data_df, seir_params):
    timesunits = seir_params.S.shape[0]
    times = np.arange(0, seir_params.S.shape[0], 1)
    S, E, I, R = solve_seir(
        S0=seir_params.S.iloc[0], 
        E0=seir_params.E.iloc[0],
        I0=seir_params.I.iloc[0],
        R0=seir_params.R.iloc[0],
                betas = get_zerostep_func(timesunits, x1),
               gamma=seir_params.gamma, alpha=alpha, mu=0, times=times
              )

    return male_metric(I+R, data_df.confirmed)


def optimize_zerostep_beta(data_df, seir_params):
    params = differential_evolution(lambda x: zerostep_beta_cost_func(x[0], data_df, seir_params), 
              bounds=[(0, 10.0)], 
                                    seed=157)
    return get_zerostep_func(data_df.shape[0], params.x[0]), params
    
    
    
    
def optimize_onestep_beta(data_df, seir_params):
    params = differential_evolution(lambda x: onestep_beta_cost_func(x[0], x[1], x[2], data_df, seir_params), 
              bounds=[(0, 1), (0, 1), (0, data_df.shape[0])], 
                                    seed=157)
    return get_step_func(data_df.shape[0], params.x[0], params.x[1], int(params.x[2])), params
    
    
    
def twostep_beta_cost_func(x1, x2, x3, middle1, middle2, data_df, seir_params):
    timesunits = seir_params.S.shape[0]
    times = np.arange(0, seir_params.S.shape[0], 1)
    S, E, I, R = solve_seir(
        S0=seir_params.S.iloc[0], 
        E0=seir_params.E.iloc[0],
        I0=seir_params.I.iloc[0],
        R0=seir_params.R.iloc[0],
                betas = get_twostep_func(timesunits, x1, x2, x3, int(middle1), int(middle2)),
               gamma=seir_params.gamma, alpha=alpha, mu=0, times=times
              )

    return male_metric(I+R, data_df.confirmed)

# def _opt_2side_func(x):
#         return twostep_beta_cost_func(x[0], x[1], x[2], x[3], x[4], data_df, seir_params)

def optimize_twostep_beta(data_df, seir_params):   
    params = differential_evolution(
        lambda x: twostep_beta_cost_func(x[0], x[1], x[2], x[3], x[4], data_df, seir_params), 
              bounds=[(0, 1), (0, 1), (0, 1), (0, data_df.shape[0]), (0, data_df.shape[0])], 
                                    seed=157)
    return get_twostep_func(data_df.shape[0], 
                            params.x[0], params.x[1], params.x[2], int(params.x[3]), int(params.x[4])), params
                            
                            
                            
                            


def draw_seir_approximation(data_df, seir_params, contry_name, beta_opt_func=optimize_onestep_beta):
    opt_beta, opt_result = beta_opt_func(data_df, seir_params)
    timesunits = data_df.shape[0]
    times = np.arange(0, timesunits, 1)
    S, E, I, R = solve_seir(
                    S0=seir_params.S.iloc[0],
                    E0=seir_params.E.iloc[0], 
                    I0=seir_params.I.iloc[0], 
                    R0=seir_params.R.iloc[0], 
    #                 betas = italy_params.beta*0.94,
                    betas=opt_beta,
                    gamma=seir_params.gamma, 
                    alpha=alpha, 
                    mu=0, 
                    times=times
              )

    ax = plot_ir(I, R, data_df.confirmed, contry_name, times)
    return opt_beta
    
    
def run_train_test(data_df, contry_name, N, cut_koef=500, dot_number=None, test_fold_size=1, beta_opt_func=optimize_zerostep_beta):
    train_data = data_df.iloc[:-test_fold_size, :]
    if dot_number is None:
        train_data = cut_confirmed(train_data, cut_koef)
    else:
        train_data = cut_confirmed_to_number(train_data, number=dot_number)
    
    train_params = compute_scalar_params(train_data, N)
    
    opt_beta, opt_result = beta_opt_func(train_data, train_params)
    
    test_data = pd.concat([train_data, data_df.iloc[-test_fold_size:, :]])
    timesunits = test_data.shape[0]
    times = np.arange(0, timesunits, 1)
    S, E, I, R = solve_seir(
                    S0=train_params.S.iloc[0],
                    E0=train_params.E.iloc[0], 
                    I0=train_params.I.iloc[0], 
                    R0=train_params.R.iloc[0], 
                    betas=opt_beta,
                    gamma=train_params.gamma, 
                    alpha=alpha, 
                    mu=0, 
                    times=times
              )
    
    fig = plt.figure(figsize=(15, 4))
    axis = fig.subplots(1, 2)
    
    plot_ir(I, R, test_data.confirmed, contry_name, times, axis[0])
    xticks = times[::4]
    xlabels = train_data.index[::4].map(lambda dt: f'{dt.year}-{dt.month:02d}-{dt.day:02d}')
    axis[0].set_xticks(xticks)
    axis[0].set_xticklabels(xlabels)
    axis[0].tick_params(axis='x', rotation=90)
    
    axis[1].plot(opt_beta[:train_data.shape[0]])    
    axis[1].set_xticks(xticks)
    axis[1].set_xticklabels(xlabels)
    axis[1].tick_params(axis='x', rotation=90)

    return opt_beta, train_params, S, E, I, R
    
alpha=1/5.1



def train(country_name, data, N, dot_number=8):
#     try:
#         country_stat = data[country_name]
        train_data = cut_confirmed_to_number(data, number=dot_number)
#         N=rus_regions[rus_regions.csse_province_state==country_name]['population'].iloc[0]
    
        train_params = compute_scalar_params(train_data, N)
        opt_beta, opt_result = optimize_zerostep_beta(train_data, train_params)
#         countries_seir_params[country_name] = train_params, opt_beta, train_data, opt_result
        return train_params, opt_beta, train_data, opt_result

        print(f'\t Training completed. Cost function value = {opt_result.fun}')



prediction_period = 300

def predict(model_params):
    betas = model_params[1]
    betas = np.hstack([betas, np.ones(prediction_period*2)*betas[-1]])
    new_index = model_params[2].index.tolist()
    new_index = new_index + [new_index[-1] + pd.Timedelta(f'{n} days') for n in range(1, prediction_period+1)]
    times = np.arange(0, len(new_index), 1)
    
    I0=model_params[0].I.iloc[0]
    R0=model_params[0].R.iloc[0]
    if I0 == 0:
        I0=1
        if R0>0:
            R0-=1
        
    S, E, I, R = solve_seir(
                S0=model_params[0].S.iloc[0],
                E0=model_params[0].E.iloc[0], 
                I0=I0, 
                R0=R0, 
                betas=betas,
                gamma=model_params[0].gamma, 
                alpha=alpha, 
                mu=0, 
                times=times
          )
    
    return pd.DataFrame({'predicted_confirmed': I+R}, index=new_index)


def train_all(rus_regions, countries, rus_directory, country_directory, dot_number=2):
    rus_data = {}
    for province in rus_regions.csse_province_state:
        rus_data[province] = read_data_for_country('Russia', province, directory=rus_directory)

    country_data = {}
    for name in countries.ccse_name:
        try:
            country_data[name] = read_timeseries_for_country(name, directory=country_directory)
        except:
            print(name)

    countries_seir_params = {}
    counter = 0
    for country_name, country_stat in tqdm(country_data.items(), total=len(country_data)):
        counter+=1
        print(f'{counter}. Training is starting: {country_name}')
        countries_seir_params[country_name] = train(country_name, country_stat, 
            N=countries[countries.ccse_name==country_name]['population'].iloc[0], dot_number=dot_number)
        
    rus_seir_params = {}
    counter = 0
    for province, data in tqdm(rus_data.items(), total=len(rus_data)):
        counter+=1
        print(f'{counter}. Training is starting: {province}')
        rus_seir_params[province] = train(province, data, \
            N=rus_regions[rus_regions.csse_province_state==province]['population'].iloc[0], dot_number=dot_number)
            
            
    rus_subm = pd.DataFrame({'date': [], 'region': [], 'prediction_confirmed': [], 'prediction_deaths': []})
    for name, seir in rus_seir_params.items():
        pred = predict(rus_seir_params[name])
        new_subm = pd.DataFrame({
            'date': pred.index,
            'region': rus_regions[rus_regions.csse_province_state==name].iso_code.iloc[0],
            'prediction_confirmed': pred.predicted_confirmed,
            'prediction_deaths': 0
        })
        rus_subm = pd.concat([rus_subm, new_subm])
        
        
    country_subm = pd.DataFrame({'date': [], 'region': [], 'prediction_confirmed': [], 'prediction_deaths': []})
    for name, seir in countries_seir_params.items():
        pred = predict(countries_seir_params[name])
        new_subm = pd.DataFrame({
            'date': pred.index,
            'region': countries[countries.ccse_name==name].iso_alpha3.iloc[0],
            'prediction_confirmed': pred.predicted_confirmed,
            'prediction_deaths': 0
        })
        country_subm = pd.concat([country_subm, new_subm])
        
        
    subm = pd.concat([rus_subm, country_subm])
    subm.prediction_confirmed = subm.prediction_confirmed.astype(int)
    subm.prediction_deaths = subm.prediction_deaths.astype(int)

    return subm