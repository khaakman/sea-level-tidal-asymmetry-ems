#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import hatyan
import datetime
station = 'delfzijl'

def load_data(fn):
    data = pd.read_csv(fn, usecols=['MEETAPPARAAT_OMSCHRIJVING', 'WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'NUMERIEKEWAARDE', 'WAARDEBEPALINGSMETHODE_OMSCHRIJVING', 'GROEPERING_OMSCHRIJVING', 'OPDRACHTGEVENDE_INSTANTIE'],  delimiter=';', encoding="ISO-8859-1", parse_dates=[['WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)']])
    data = data.set_index('WAARNEMINGDATUM_WAARNEMINGTIJD (MET/CET)')
    data = data[data['WAARDEBEPALINGSMETHODE_OMSCHRIJVING'] != 'HW en LW uit 1 min. waterhoogten gefilterd uit 10 min. gem.']
    data = data[data['WAARDEBEPALINGSMETHODE_OMSCHRIJVING'] == 'Rekenkundig gemiddelde waarde over vorige 5 en volgende 5 minuten']
    data = data[data['NUMERIEKEWAARDE'] != 999999999]
    data = data[data['MEETAPPARAAT_OMSCHRIJVING'] == 'Vlotter']
    data.index = pd.to_datetime(data.index, format='%d-%m-%Y %H:%M:%S')
    data = data[['NUMERIEKEWAARDE']] 
    data = data.rename(columns={'NUMERIEKEWAARDE': 'values'})
    data.values = data.values / 100 #cm to m
    
    data = data[data.index < datetime.datetime(year=2023, month=1, day=1)]
    
    completeness = data["values"].notna().resample("YE").sum().to_frame("valid")
    completeness["expected"] = completeness.index.year.map(
        lambda y: 366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365
        ) * 24 * 6
    completeness["completeness_%"] = 100 * completeness["valid"] / completeness["expected"]
    print(completeness)
    
    print('Mean completeness: ', round(100*(np.sum(completeness['valid']) / np.sum(completeness['expected'])),1))
    return data  

def main():
    const_list = hatyan.get_const_list_hatyan('year')
    const_list.remove('MNO5') #aliasing 
    fn = '../../data/RWS/delfzijl_1987_2025_10min.csv'
    data = load_data(fn)
    
    hatyan_result = hatyan.analysis(data, const_list, nodalfactors=False, 
                                    fu_alltimes=False, analysis_perperiod='Y',
                                    return_allperiods=True)
    amps = hatyan_result[1]['A']
    phases = hatyan_result[1]['phi_deg']
    for yr in np.arange(1987, 2023):
        amps_yr = amps[str(yr)]
        phases_yr = np.radians(phases[str(yr)])
        savefolder = '../../processed_time_series/tide_gauges/{}/'.format(station)
        os.makedirs(savefolder, exist_ok=True)
        np.save(savefolder+'amps_{}_{}.npy'.format(station, yr), amps_yr[1:])
        np.save(savefolder+'phases_{}_{}.npy'.format(station, yr), phases_yr[1:])
    return


if __name__ == '__main__':
    main()