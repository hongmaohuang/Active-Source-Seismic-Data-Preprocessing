# %%

import obspy
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from obspy import Trace
from obspy.signal.cross_correlation import correlate
from obspy.io.sac import attach_paz
from obspy.signal.invsim import corn_freq_2_paz
from obspy.core.stream import Stream
from scipy import signal
from datetime import datetime, timedelta
from matplotlib.dates import date2num
import shutil
from obspy.core import UTCDateTime
from scipy.signal import hilbert

# ===================================================== # 
def PWS(st, v, sm=False, sl=15):
    m = len(st)
    n = st[0].stats.npts
    dt = st[0].stats.delta
    t = np.arange(n) * dt
    c = np.zeros(n, dtype=complex)
    for i, tr in enumerate(st):
        h = hilbert(tr.data)
        c += h/abs(h)
    c = abs(c/m)
    if sm:
        operator = np.ones(sl) / sl
        c = np.convolve(c, operator, 'same')
    stc = st.copy()
    stc.stack()
    tr = stc[0]
    tr.data = tr.data*c**v
    return tr

# ================ 參數設定（By case） ================ #
array = 'hongcailin'
#ilan, hongcailin
component = 'DPZ'
# 'DPE', 'DPN', 'DPZ'
Line_AS = 'SV1'
# 'SV1', 'SV2', 'SV_South', 'SV_North'
if Line_AS == 'SVA_North':   
    AS_table_file = 'N65neo_6to80Hz_12s_D75.custom'
    shottime_table = Line_AS + '_shottime(6~80hz).csv'
else: 
    AS_table_file = 'N65neo_6to96Hz_12s_D75.custom'
    shottime_table = Line_AS + '_shottime(6~96hz).csv'


freqmin = 1
freqmax = 30
#濾波

shottime_table_path = '/home/hmhuang/Work/Research_Assistant/AS_HCL/shottime_table/'
# ================ 參數設定（無需更動） ================ #
project_path = '/home/hmhuang/Work/Research_Assistant/AS_HCL/'
esp_path = '/raid2/ILAN2022/PZs/'
AS_table_path = 'AS_table/'
resp_path = '/raid2/ILAN2022/PZs/'
output_path = '/raid2/ILAN2022/CCF/new_one/'
if array=='ilan':
    stations_file = '/raid2/ILAN2022/ILAN_sta.csv'
    sac_path_head = '/raid2/ILAN2022/Sorted_oneday_ILAN_array/'
else:
    stations_file = '/raid2/ILAN2022/stations.csv'
    sac_path_head = '/raid2/ILAN2022/Sorted_oneday/'


AS_time_once = 12
sensitivity_resp = 306846
corner_freq = 5
damping = 0.7
gain = 1
pre_filt = (0.05, 0.1, 110, 125)
once_sec = 30
#取幾秒做CCF



table_shottime = pd.read_table(shottime_table_path + shottime_table, sep=',')
all_times_point = table_shottime.PointNumber
all_as_point = []
for q in all_times_point:
    if q not in all_as_point:
        all_as_point.append(q)
all_as_point = np.array(all_as_point)
sta_all = pd.read_csv(stations_file).Station
for sta_mkdir in sta_all:
    if os.path.exists(output_path + 'test_CC_time_check/' +  Line_AS + '/' + component):
        continue
    else:
        os.makedirs(output_path + 'test_CC_time_check/' + Line_AS + '/' + component)
'''
for sta_mkdir in sta_all:
    if os.path.exists(output_path + str(sta_mkdir)):
        shutil.rmtree(output_path + str(sta_mkdir))
        os.makedirs(output_path + str(sta_mkdir))
    else:
        os.makedirs(output_path + str(sta_mkdir))
'''
# ================ 震源函數資料格式修正 ================ #
table_AS = pd.read_table(project_path + AS_table_path + AS_table_file, delimiter=' ', skiprows = 44)
new_AS_table = table_AS.rename(columns={'0.00000000':'data', 'Unnamed: 1':'one', 'Unnamed: 2':'two', 'Unnamed: 3':'three', 'Unnamed: 4':'four', 'TRUE':'true'})
new_AS_table.loc[-1] = [float(table_AS.keys()[0]), 'NaN', 'NaN', 'NaN', 'NaN', 'True']
new_AS_table.index = new_AS_table.index + 1
new_AS_table = new_AS_table.sort_index()
#把資料第一行被當作header修正

for index, sta in enumerate(sta_all):
    for j in range(0, len(all_as_point)):
        if os.path.exists(output_path + 'test_CC_time_check/' + Line_AS + '/' + component + '/' + Line_AS + '_' + str(sta) + '_' + str(all_as_point[j]) + '.sac'):
            print(output_path + 'test_CC_time_check/' + Line_AS + '/' + component + '/' + Line_AS + '_' + str(sta) + '_' + str(all_as_point[j]) + '.sac exsisted')
            continue
        #print(str(j) + '/' + str(len(all_as_point)))
        AS_point = all_as_point[j]
        loc_as_number = table_shottime[table_shottime.PointNumber == AS_point].index
        #fig = plt.figure()
        tr_stack_all = []
        st_stack_all = Stream()
        print(Line_AS + '_' + str(sta) + '_' + str(all_as_point[j])  + '.sac' + ' ' + component  +' ----' + str(j) + ' in ' + str(len(all_as_point)) + '------------------' + str(index) + '/' + str(len(sta_all)) + ' is processing!!')
        for i in range(0,len(loc_as_number)):
            
            #print(str(sta) + ' ' + str(j) + str(i))
            time_trans = datetime.strptime(np.array(table_shottime.Date[loc_as_number])[i], '%Y/%m/%d')
            datethistime = datetime.strftime(time_trans, '%Y%m%d')
            year = datethistime[0:4]
            month = datethistime[4:6]
            day = datethistime[6:8]
            sac_path = sac_path_head + year + month + day + '/'
            shottime_assign = np.array(table_shottime.Time[loc_as_number])[i]
            if shottime_assign[0]==' ':
                shottime_assign = shottime_assign[1:len(shottime_assign)]
            time_trans = datetime.strptime(shottime_assign,'%H:%M:%S.%f')
            time_correct = time_trans - timedelta(seconds=8*60*60)
            shottime_assign = datetime.strftime(time_correct,'%H:%M:%S.%f')
            #print(shottime_assign)
            date_str = year + month + day 
            date_obj = datetime.strptime(date_str, "%Y%m%d") 
            doy_DOY = date_obj.timetuple().tm_yday  

        # ================ 抓震測時間 ================ #
            all_number_in_point = []
            sac_filename = str(sta) + '.FM.00.' + component + '.' + year + '.' +  str(doy_DOY) + '.' + year + month + day + '.SAC'
            #print(sac_filename)
            # ================ 製作震源函數MSEED檔案 ================ #
            y = new_AS_table.data
            tr = Trace(np.array(y))
            tr.stats.sampling_rate = (len(table_AS) + 1) / AS_time_once
            if i == 0:
                save_first_vib = year + '-' + month + '-' + day + 'T' + shottime_assign[0:2] + ':' + shottime_assign[3:5] + ':' + shottime_assign[6:8] + '.' + shottime_assign[9:15] + 'Z'
            tr.stats.starttime = year + '-' + month + '-' + day + 'T' + shottime_assign[0:2] + ':' + shottime_assign[3:5] + ':' + shottime_assign[6:8] + '.' + shottime_assign[9:15] + 'Z'
            tr.write('hypof_' + component + '.mseed', format='MSEED')
            # ================ 讀取sac並裁切時間 ================ #
            try:
                st = obspy.read(sac_path+sac_filename)
                  
            except:
                #print(sac_path+sac_filename)
                continue
            st_trim = st.copy()
            st_trim_filter = st.copy()
            # 先不用濾波
            #st_trim_filter = st_trim.filter('bandpass', freqmin=6.0, freqmax=80.0)
            starttime = UTCDateTime(year + '-' + month + '-' + day + 'T00:00:00.000000Z') + int(shottime_assign[0:2])*60*60 + int(shottime_assign[3:5])*60 + float(shottime_assign[6:15])
            endtime = starttime + 8
            #endtime = starttime + 20
            #st_trim_filter.trim(starttime=starttime - 20, endtime=endtime)
            st_trim_filter.trim(starttime=starttime-2, endtime=endtime)
            try:
                tr_sac = st_trim_filter[0]

            except:
                continue
            # ================ 去除儀器響應（地震訊號）================ #
            attach_paz(tr_sac, resp_path + component)
            paz_1hz = corn_freq_2_paz(corner_freq, damp=damping)  
            paz_1hz['sensitivity'] = sensitivity_resp
            paz_1hz['gain'] = gain
            st_trim_filter.simulate(paz_remove=paz_1hz, pre_filt=pre_filt)

            # ================ 訊號基本處理（地震訊號）================ #
            tr_sac.detrend(type='demean')  
            tr_sac.detrend(type='linear')
            tr_sac.taper(max_percentage=0.05, type='cosine', max_length=len(tr[0].data), side='both')
            st_trim_filter_y = tr_sac.data

            # ================ 震源函數讀取及重新取樣 ================ #
            st_hypof = obspy.read('hypof_' + component + '.mseed')
            st_hypof.resample(tr_sac.stats.sampling_rate) 
            tr_hypof = st_hypof[0]
            #print(tr_hypof.stat.starttime)
            #print(tr_sac.stat.starttime)
            # ================ Cross-Correlation ================ #
            #print(tr_hypof.stats.starttime)
            #print(tr_sac.stats.starttime)
            corr = signal.correlate(tr_sac.data, tr_hypof.data)
            lags = signal.correlation_lags(len(tr_sac.data), len(tr_hypof.data))
            corr /= np.max(corr)
            tr_y_cc = Trace(np.array(corr))
            tr_y_cc.stats.sampling_rate = 250
            #tr_y_cc = tr_y_cc.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
            sec_ccf = lags/250
            #plt.plot(sec_ccf, tr_y_cc.data+(i+1)*1, linewidth=0.3, c='k')
            #tr_stack_all.append(tr_y_cc.data)
            st_stack_all.append(tr_y_cc)
            
            #plt.show
            if i==0:
                t_sac = tr_sac.stats.starttime
                t_hypo = tr_hypof.stats.starttime
        if len(st_stack_all) == 0:
            continue
        else:
            try:
                ststst_stack = PWS(st_stack_all, 2)              
                trtrtr_stack = ststst_stack.data
                idx_A = 0
                idx_B = int((t_sac - t_hypo) * tr_hypof.stats.sampling_rate)
                idx_C = np.where(lags == (idx_B - idx_A))[0][0]
                new_trtrtr_stack = trtrtr_stack[idx_C : len(trtrtr_stack)]
                new_seccc = sec_ccf[idx_C : len(trtrtr_stack)]
                trtrtr_stack_new_correct = new_trtrtr_stack[np.where(sec_ccf[idx_C : len(trtrtr_stack)]==0)[0][0]:len(new_seccc)]
                stack_trace = Trace(trtrtr_stack_new_correct)
                #print(stack_trace.times())
            except:
                continue
        stack_trace.stats.starttime = UTCDateTime(save_first_vib) - 4
        #print(stack_trace.stats.starttime)
        stack_trace.stats.sampling_rate = 250
        stack_trace.write(output_path + 'test_CC_time_check/' +  Line_AS + '/' + component + '/' + Line_AS + '_' + str(sta) + '_' + str(all_as_point[j]) + '.sac')
        #print(output_path + 'test_CC_time_check/' +  Line_AS + '/' + component + '/' + Line_AS + '_' + str(sta) + '_' + str(all_as_point[j]) + '.sac')
        #print(output_path + Line_AS + 'test_CC_time_check/' + component + '/' + Line_AS + '_' + str(sta) + '_' + str(all_as_point[j]) + '.sac is processed.')
        os.remove('hypof_' + component + '.mseed')

# %%