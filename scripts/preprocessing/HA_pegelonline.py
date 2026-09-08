#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import hatyan

stations = ['borkumfischerbalje', 'borkumsudstrand', 'dukegat', 'knock', 'emshorn', 'pogum', 'emdenneueseeschleuse']

def load_data(fn):
    data = pd.read_csv(fn, delimiter=',', index_col=0)
    data.index = pd.to_datetime(data.index, format ='%Y-%m-%d %H:%M:%S')
    return data

def main():
    const_list = hatyan.get_const_list_hatyan('year')
    const_list.remove('MNO5') #aliasing 
    for i, station in enumerate(stations):
        print('#### {} ####'.format(station))
        fn = '../../data/pegelonline_preprocessed/{}.csv'.format(station)
        data = load_data(fn)

        # hatyan_df = hatyan_df[~hatyan_df.index.duplicated(keep="first")]
        hatyan_result = hatyan.analysis(data, const_list, nodalfactors=False, 
                                        fu_alltimes=False, analysis_perperiod='Y',
                                        return_allperiods=True)

        amps = hatyan_result[1]['A']
        phases = hatyan_result[1]['phi_deg']
        for yr in np.arange(2000, 2023):
            amps_yr = amps[str(yr)]
            phases_yr = np.radians(phases[str(yr)])
            savefolder = '../../processed_time_series/tide_gauges/{}/'.format(station)
            os.makedirs(savefolder, exist_ok=True)
            np.save(savefolder+'amps_{}_{}.npy'.format(station, yr), amps_yr[1:])
            np.save(savefolder+'phases_{}_{}.npy'.format(station, yr), phases_yr[1:])

    return


if __name__ == '__main__':
    main()