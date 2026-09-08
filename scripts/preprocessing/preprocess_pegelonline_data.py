#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from datetime import timezone, timedelta

folder = '../../data/pegelonline_raw/'
filenames = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(folder) for f in filenames if f.endswith('.csv')]

def load_data(fn):
    data = pd.read_csv(fn, delimiter=';')
    data = data.set_index('timestamp')
    data.index = pd.to_datetime(data.index, format ='%Y-%m-%d %H:%M')
    data.index.name = 'datetime'
    data = data.rename(columns={'value':'values'})
    data['values'] = data['values'] / 100 #convert cm to m
    
    ## Pegelonline data still has daylight saving time jumps, next four lines remove it.
    data.index = data.index.tz_localize(
        'Europe/Amsterdam', nonexistent='shift_forward', ambiguous='infer')
    data.index = data.index.tz_convert('UTC')
    data.index = data.index.tz_convert(timezone(timedelta(hours=1))) #fix to UTC+1
    data.index = data.index.tz_localize(None)
    return data

def main():
    for fn in filenames:
        data = load_data(fn)
        station = fn.split('/')[-1].split('.')[0]
        savename = '../../data/pegelonline_preprocessed/{}.csv'.format(station)
        data.to_csv(savename)
        print('saved: ', savename)
    return

if __name__ == '__main__':
    main()