import pandas as pd
import os


def Temper_Precp():
    """Process open data, then output a `pandas.DataFrame` with columns `Temperature` and `Precp`"""
    res = []
    for file in os.listdir(os.path.realpath('.') + r'/opendata/'):
        if '臺北' in file:
            data = pd.read_csv(os.path.join(os.path.realpath('.') + r'/opendata/', file), header=1)
            res.append(data.loc[:, ['StnPresMaxTime', 'Temperature', 'Precp']])
    result = pd.concat(res, axis=0, ignore_index=True)
    result['StnPresMaxTime'] = pd.to_datetime(result['StnPresMaxTime'], format='%Y-%m-%d %H:%M').dt.date
    result['Precp'] = result['Precp'].str.replace('T', '0.1').astype('float')
    return result.set_index('StnPresMaxTime')


Temper_Precp = Temper_Precp()
