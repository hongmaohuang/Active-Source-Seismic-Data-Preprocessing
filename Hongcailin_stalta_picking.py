# %%
# ============================ #
#            Picking           #
# ============================ #
#
# 本程式輸出tt_PS， 
# 若需要evt_PS以及sta_PS，
# 可在第二層迴圈中修改
#（已寫好只需解除註解）



import numpy as np
import scipy.io
def signaltonoise(a, axis=0, ddof=0):
    return np.where(sd == 0, 0, m/sd)

#改成峰值/雜訊rms

# ================ Modules Loading ================ #
import obspy
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
from datetime import datetime as dt
from obspy.signal.trigger import ar_pick
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import z_detect
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import plot_trigger
import os
import warnings

# ================ Parameters（By case） ================ #
version = 'V14'
sac_with_t_output = '/raid2/ILAN2022/sac_with_t1/' + version + '/'
ALL_Line = ['SV1', 'SV2', 'SVA_North', 'SVA_South']
ALL_Line = ['SVA_South']


siesmic_profile_output = '/home/hmhuang/Work/Research_Assistant/picking/siesmic_profile_output_' + version + '/'
tt_PS_final = pd.DataFrame()
for Line_AS in ALL_Line:
    shottime_table_path = '/home/hmhuang/Work/Research_Assistant/AS_HCL/shottime_table/'
    if Line_AS == 'SVA_North':
        shottime_table = Line_AS + '_shottime(6~80hz).csv'
    else:    
        shottime_table = Line_AS + '_shottime(6~96hz).csv'
    output_path = '/home/hmhuang/Work/Research_Assistant/picking/' + version + '/'
    table_shottime = pd.read_table(shottime_table_path + shottime_table, sep = ',')
    ALL_AS_point = table_shottime.PointNumber
    ALL_AS_point.drop_duplicates(keep='first', inplace=True)
    tt_PS_thisline = pd.DataFrame()
    for order_now, AS_point in enumerate(ALL_AS_point):
        #if order_now >0 :
        #    continue
        # 如果只需要跑evt_PS以及sta_PS
        print(f"Processing {AS_point}, {order_now+1}/{len(ALL_AS_point)}, {Line_AS}")
        # 震源號碼（reference to MAP）
        method = 'recursive_sta_lta'
        # classic_sta_lta, recursive_sta_lta, ar_pick, z_detect
        cft_therh = 2
        # STA/LTA ratio
        freq = [15, 30]
        # freq. for band pass
        lta_p = 0.3
        sta_p = 0.03
        #lta_s = 0.5
        #sta_s = 0.1
        # Length of LTA and STA for the P and S arrival in seconds.

        #m_p = 5
        #m_s = 5
        # Number of AR coefficients for the P and S arrival.
        #l_p = 0.01
        #l_s = 0.1
        # Length of variance window for the P and S arrival in seconds.

        filled = 'no'
        # randomS, zero, no, self
        mu, sigma = 0, 0.085 # for random noise



        # ================ plot parameters ================ #
        #ylim_dist = [2000, 3000]
        ylim_dist = 'no'
        # Adjusting ylim on the plot
        # set 'no' or an interval
        l_redbar = 50
        # Adjusting the red bar on the plot
        xlim_p = [2, 3.5]
        # setting xlim on the plot

        # ================ useless for now ================ #
        # length_self = 0.1 # for self
        # do not more than 0.1
        # times_self = 1 # for self
        # 將自身訊號放大幾倍

        # ================ No need to change here ================ #
        snr_wind = [0, 2000]
        # snr computation windows size in samples
        component = ['DPZ', 'DPE', 'DPN']
        sac_filepath = '/raid2/ILAN2022/CCF/from_PWS_stacking/hongcailin_array/'
        stations_file = '/raid2/ILAN2022/stations.csv'
        shottime_table_path = '/home/hmhuang/Work/Research_Assistant/AS_HCL/shottime_table/'
        if Line_AS == 'SVA_North':
            shottime_table = Line_AS + '_shottime(6~80hz).csv'
        else:    
            shottime_table = Line_AS + '_shottime(6~96hz).csv'
        output_path = '/home/hmhuang/Work/Research_Assistant/picking/' + version + '/'

        # ================ Coordinate of the Source ================ #
        table_shottime = pd.read_table(shottime_table_path + shottime_table, sep = ',')
        loc_assign = np.where(table_shottime.PointNumber==AS_point)
        lon_source = table_shottime.Lon[loc_assign[0][0]]
        lat_source = table_shottime.Lat[loc_assign[0][0]]

        # ================ Calculation ofr distance between stations and the source ================ #
        name_sta_o = pd.read_csv(stations_file).Station
        lon_sta_o = pd.read_csv(stations_file).Lon
        lat_sta_o = pd.read_csv(stations_file).Lat
        all_dist = (((lon_sta_o - lon_source) ** 2 + (lat_sta_o - lat_source) ** 2) ** 0.5)/180*math.pi*6378137
        dict_sta_dist = {'Sta': name_sta_o, 'Dist': all_dist, 'Lon': lon_sta_o, 'Lat': lat_sta_o}
        df_sta_dist = pd.DataFrame(data=dict_sta_dist)
        all_file_name_Z = []
        all_file_name_E = []
        all_file_name_N = []
        for i in name_sta_o:
            all_file_name_Z.append(sac_filepath + Line_AS + '/' + component[0] + '/' + Line_AS + '_' + str(i) + '_' + str(AS_point) + '.sac')
            all_file_name_E.append(sac_filepath + Line_AS + '/' + component[1] + '/' + Line_AS + '_' + str(i) + '_' + str(AS_point) + '.sac')
            all_file_name_N.append(sac_filepath + Line_AS + '/' + component[2] + '/' + Line_AS + '_' + str(i) + '_' + str(AS_point) + '.sac')
        df_sta_dist['Path_Z'] = all_file_name_Z
        df_sta_dist['Path_E'] = all_file_name_E
        df_sta_dist['Path_N'] = all_file_name_N
        df_sta_dist.sort_values(by='Dist', inplace=True)
        df_sta_dist = df_sta_dist.reset_index(drop=True)
        
        # ================ making sta_demo ================

        file_sta_o = pd.read_csv(stations_file)
        file_sta = file_sta_o.rename(columns={'H(m)': 'H'})
        file_sta_new = pd.DataFrame()
        file_sta_new['Sta'] = file_sta.Station
        file_sta_new['Lat'] = file_sta.Lat
        file_sta_new['Lon'] = file_sta.Lon
        file_sta_new['H'] = file_sta.H
        file_sta_new.to_csv(output_path + 'sta_PS', sep=' ', index=False, header=False)
        
        # ================ making evt_demo ================
        table_shottime = pd.read_table(shottime_table_path + shottime_table, sep = ',')
        new_shottable = pd.DataFrame()
        all_datetime = table_shottime.Date
        new_date_str = []
        delta_tt = datetime.timedelta(hours=8)
        for i in all_datetime:
            date_obj = dt.strptime(i, "%Y/%m/%d")
            new_date_str.append(date_obj.strftime("%Y%m%d"))

        all_time = table_shottime.Time
        new_time_str = []
        for j in all_time:
            new_time_str.append(dt.strptime(j.strip(), '%H:%M:%S.%f') - delta_tt)


        new_time_str_final = []
        for k in range(len(new_time_str)):
            sec = new_time_str[k].second
            microsec = new_time_str[k].microsecond
            delta_time_sec = datetime.timedelta(seconds=sec, microseconds=microsec)
            total_seconds = delta_time_sec.total_seconds()
            x = total_seconds
            n = 5
            p = 3
            formatted_sec = '{:0>{n}.{p}f}'.format(x, n=n, p=p)
            formatted_sec_without_dot = formatted_sec.replace('.', '')
            minute = new_time_str[k].minute
            x = minute
            n = 2
            formatted_minute = '{:0>{n}}'.format(minute, n=n)
            if len(formatted_sec_without_dot) == 4:
                formatted_sec_without_dot = str(0) + formatted_sec_without_dot
            finnal_time = str(0) + str(new_time_str[k].hour) + formatted_minute + formatted_sec_without_dot
            new_time_str_final.append(finnal_time)


        new_shottable['Date'] = new_date_str
        new_shottable['Time'] = new_time_str_final
        new_shottable['Lat'] = table_shottime.Lat
        new_shottable['Lon'] = table_shottime.Lon
        new_shottable['Depth'] = table_shottime.Altitude * -0.001
        useless_array = [0] * len(table_shottime)
        for i in range(len(useless_array)):
            useless_array[i] = 0
        new_shottable['mag'] = useless_array
        new_shottable['junk1'] = useless_array
        new_shottable['junk2'] = useless_array
        new_shottable['junk3'] = useless_array
        new_pointnumber_id = []
        for i in range(len(table_shottime.PointNumber)):
            if Line_AS == 'SV1':
                new_pointnumber_id.append(str(1) + str(table_shottime.PointNumber[i])) 
            if Line_AS == 'SV2':
                new_pointnumber_id.append(str(2) + str(table_shottime.PointNumber[i])) 
            if Line_AS == 'SVA_North':
                new_pointnumber_id.append(str(3) + str(table_shottime.PointNumber[i])) 
            if Line_AS == 'SVA_South':
                new_pointnumber_id.append(str(4) + str(table_shottime.PointNumber[i])) 
        new_shottable['id'] = new_pointnumber_id
        new_shottable.drop_duplicates(subset='id', keep='first', inplace=True)
        new_shottable.to_csv(output_path + 'evt_PS_' + Line_AS, sep=' ', index=False, header=False)
        # ================ Plot ================ #
        fig = plt.figure(1, figsize=(8, 10))
        randoms = np.random.normal(mu, sigma, 250)
        picked_p_tt = []
        snr = []
        picked_p_dist = []
        picked_p_sta = []
        picked_p_sta_lon = []
        picked_p_sta_lat = []
        picked_p_tt_all = []
        picked_p_dist_all = []
        #print(df_sta_dist)
        for i in range(len(df_sta_dist)):
            try:
                stE = obspy.read(df_sta_dist.Path_E[i])
                #stE.filter('bandpass', freqmin=freq[0], freqmax=freq[1])
                stE.filter('highpass', freq=15)
                stN = obspy.read(df_sta_dist.Path_N[i])
                #stN.filter('bandpass', freqmin=freq[0], freqmax=freq[1])
                stN.filter('highpass', freq=15)
                stZ = obspy.read(df_sta_dist.Path_Z[i])
                #stZ.filter('bandpass', freqmin=freq[0], freqmax=freq[1])
                stZ.filter('highpass', freq=15)
                stE.trim(stE[0].stats.starttime, stE[0].stats.endtime)   
                stN.trim(stN[0].stats.starttime, stN[0].stats.endtime)
                stZ.trim(stZ[0].stats.starttime, stZ[0].stats.endtime)
                df = stE[0].stats.sampling_rate
                trE = stE[0]
                trN = stN[0]
                trZ = stZ[0]
                a = np.asanyarray(trZ.data)
                #m = a[snr_wind[0]:snr_wind[1]].max(0)
                #sd = a[snr_wind[0]:snr_wind[1]].std(axis=0, ddof=0)
                #snr_inter = m/sd 
                #if snr_inter < SNR_threh or np.isnan(snr_inter):
                #    continue
                trE_dt = trE.data
                trN_dt = trN.data
                trZ_dt = trZ.data
                tr_times = trE.times()
                if method == 'classic_sta_lta':
                    cft = classic_sta_lta(trZ_dt, int(sta_p*df), int(lta_p*df))
                    p_pick_loc = np.where(cft == cft.max())
                    p_pick = trZ.times()[p_pick_loc][0]
                if method == 'ar_pick':
                    p_pick, s_pick = ar_pick(trZ_dt, trN_dt, trE_dt, df, freq[0], freq[1], lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s, s_pick =False)
                if method == 'recursive_sta_lta':
                    cft = recursive_sta_lta(trZ_dt, int(sta_p*df), int(lta_p*df))
                    cft = np.diff(cft)
                    p_pick_loc = np.where(cft == np.max(cft[int((lta_p*df)+1):]))
                    p_pick = trZ.times()[p_pick_loc][0] 
                if method =='z_detect':
                    cft = z_detect(trZ_dt, int(sta_p*df))
                    p_pick_loc = np.where(cft == cft.max())
                    p_pick = trZ.times()[p_pick_loc][0]

                pick_sample_point = p_pick*250
                pick_sample_point_Before = pick_sample_point - 1*250
                pick_sample_point_after = pick_sample_point + 0.2*250
                SNR_before = a[int(pick_sample_point_Before):int(pick_sample_point)].std(axis=0, ddof=0)
                SNR_after = a[int(pick_sample_point):int(pick_sample_point_after)].std(axis=0, ddof=0)
                # 這邊可能造成warning >> 關於時窗大小與其計算std的狀況
                warnings.filterwarnings("ignore")
                #print(SNR_before)
                #print(SNR_after)
                SNR_new = SNR_after/SNR_before
                snr.append(SNR_new)
                picked_p_tt_all.append(p_pick)
                picked_p_dist_all.append(df_sta_dist.Dist[i])
                
                # ============================== criteria ============================== #
                if SNR_new <= 9:
                    continue
                #if len(np.where(cft.max() > cft_therh)[0]) == 0:
                #    continue
                # 感覺用處不是很大，因為大pick有可能不是P波，用這個檢視不太準確
                if p_pick < 2 or p_pick > 7:
                    continue
                # 兩秒前根本還沒施震
                #if len(picked_p_tt) > 0 and picked_p_tt[-1] >= p_pick:
                    #continue
                # 下一個pick應該比上一個pick前面，用這一行來限制
                # 不過都要手動修了，有這種情形手動修就好    
                if len(picked_p_tt) > 0 and abs(p_pick - picked_p_tt[-1]) > 0.2:
                    continue
                #if p_pick > df_sta_dist.Dist[i]/7000+0.1+2 or p_pick < df_sta_dist.Dist[i]/7000-0.1+2:
                #    continue
                # ============================== criteria ============================== #
                picked_p_tt.append(p_pick)
                picked_p_dist.append(df_sta_dist.Dist[i])
                picked_p_sta.append(df_sta_dist.Sta[i])
                picked_p_sta_lon.append(df_sta_dist.Lon[i])
                picked_p_sta_lat.append(df_sta_dist.Lat[i])
                path_parts = df_sta_dist.Path_E[i].split("/") 
                if Line_AS == 'SV1':
                    sta_name = str(path_parts[-1][4:8])
                    source_name =  str(1) + str(path_parts[-1][9:13])
                if Line_AS == 'SV2':
                    sta_name = str(path_parts[-1][4:8])
                    source_name =  str(2) + str(path_parts[-1][9:13])
                if Line_AS == 'SVA_North':
                    sta_name = str(path_parts[-1][10:14])
                    source_name = str(3) + str(path_parts[-1][15:19])
                if Line_AS == 'SVA_South':
                    sta_name = str(path_parts[-1][10:14])
                    source_name = str(4) + str(path_parts[-1][15:19])
                stE[0].stats.sac['t1'] = p_pick 
                stN[0].stats.sac['t1'] = p_pick 
                stZ[0].stats.sac['t1'] = p_pick
                stE[0].stats.sac['dist'] = df_sta_dist.Dist[i]
                stN[0].stats.sac['dist'] = df_sta_dist.Dist[i]
                stZ[0].stats.sac['dist'] = df_sta_dist.Dist[i]
                stE[0].stats.sac['station'] = df_sta_dist.Sta[i]
                stN[0].stats.sac['station'] = df_sta_dist.Sta[i]
                stZ[0].stats.sac['station'] = df_sta_dist.Sta[i]
                stE[0].stats.startttime = stE[0].stats.starttime
                stN[0].stats.startttime = stN[0].stats.starttime
                stZ[0].stats.startttime = stZ[0].stats.starttime
                shottime_real = stE[0].stats.starttime + 2
                #stE = stE.trim(shottime_real , stE[0].stats.endtime)   
                #stN = stN.trim(shottime_real, stN[0].stats.endtime)   
                #stZ = stZ.trim(shottime_real, stZ[0].stats.endtime)   
                stE.write(sac_with_t_output + sta_name + '_' + source_name + '_DPE.sac')
                #print(sac_with_t_output + df_sta_dist.Path_E[i][40:44] + '_' + new_pointnumber_id + '_DPE.sac')
                stN.write(sac_with_t_output + sta_name + '_' + source_name + '_DPN.sac')
                stZ.write(sac_with_t_output + sta_name + '_' + source_name + '_DPZ.sac')
                #print(str(df_sta_dist.Path_N[i][40:44]) + '_' + str(new_pointnumber_id) + '_DPN.sac')
                #plt.plot([p_pick,p_pick], [np.mean(trZ_dt)*50+df_sta_dist.Dist[i]-l_redbar, np.mean(trZ_dt)*50+df_sta_dist.Dist[i]+l_redbar], 'r')
                #cft = classic_sta_lta(trZ.data, 0.05*df, 0.1*df)
                #plt.plot(tr_times, trZ_dt*50+df_sta_dist.Dist[i], 'k')
                #plt.show()
                #plt.savefig(df_sta_dist.Path_E[i][40:44] + '_' + new_pointnumber_id + '_DPE.png')
                #plt.plot([cft[-1],cft[-1]], [np.mean(trZ.data)*50+df_sta_dist.Dist[i]-l_redbar, np.mean(trZ.data)*50+df_sta_dist.Dist[i]+l_redbar], 'r')
                #plt.xlim([0,1])
            except:
                continue
            #if i == 0:
            plt.plot(stZ[0].times(), stZ[0].data*100+df_sta_dist.Dist[i], 'k', linewidth=0.7)
            plt.plot([p_pick,p_pick], [np.mean(trZ_dt)*50+df_sta_dist.Dist[i]-l_redbar, np.mean(trZ_dt)*50+df_sta_dist.Dist[i]+l_redbar], 'r')
            #''' 要畫phasenet comparisonㄉ
            #data = np.genfromtxt('/raid1/share/for_HM_2/phasenet_result/sta_lst_tmp', skip_header=1)
            #ptime = data[:, 4]
            #stn_evt_distance = data[:, 8]
            #valid_indices = (ptime != -12345) & (stn_evt_distance != -12345)
            #ptime = ptime[valid_indices]
            #stn_evt_distance = stn_evt_distance[valid_indices]
            #plt.plot([ptime, ptime], [stn_evt_distance*1000-l_redbar, stn_evt_distance*1000+l_redbar], 'g')
            #'''
            plt.text(10.4, np.mean(trZ_dt*100+df_sta_dist.Dist[i]), df_sta_dist.Sta[i])  
            plt.rcParams['font.sans-serif'] =  'Nimbus Roman'
            plt.title('Seismic Reflection Profile (Hongcailin Array)\n' + Line_AS + ': ' + str(AS_point))
            plt.ylabel('Distance [m]')
            plt.xlabel('Time [sec]')
            #plt.ylim([0, 14000])
            plt.xlim([2, 10])
            #''' 要畫phasenet comparisonㄉ
            #plt.xlim([2, 4])
            #plt.savefig(siesmic_profile_output + Line_AS + '_' + str(AS_point) + '_new.png', dpi=300)
            #'''
            plt.savefig(siesmic_profile_output + Line_AS + '_' + str(AS_point) + '.png', dpi=300)

        plt.close()
        plt.scatter(picked_p_tt_all, picked_p_dist_all, c = snr, cmap='jet')
        cbar = plt.colorbar()
        cbar.set_label('SNR')

        # Set the range of the color bar
        cbar.mappable.set_clim(0,15)
        plt.rcParams['font.sans-serif'] =  'Nimbus Roman'
        plt.savefig(siesmic_profile_output + Line_AS + '_' + str(AS_point) + '_snr_plot_' +'.png', dpi=300)
        plt.close()
        data = snr 
        bins = np.arange(0, 200, 3)
        hist, _ = np.histogram(data, bins=bins)
        plt.bar(bins[:-1], hist, width=3, align='edge', edgecolor='black')
        plt.xticks(np.arange(0, 200, 3), rotation=45, fontsize=5)
        plt.ylim([0, 150])
        plt.xlabel('SNR')
        plt.ylabel('Counts')
        plt.rcParams['font.sans-serif'] =  'Nimbus Roman'
        plt.savefig(siesmic_profile_output + Line_AS + '_' + str(AS_point) + '_snr_plot_count' +'.png', dpi=300)
        plt.close()
        
        #plot_trigger(trZ, cft, 1.2, 0.5, show=True)
        if len(picked_p_tt)==0:
            print('nothing happend!\n')
            continue
        if Line_AS == 'SV1':
            new_pointnumber_id = str(1) + str(AS_point)
        if Line_AS == 'SV2':
            new_pointnumber_id = str(2) + str(AS_point)
        if Line_AS == 'SVA_North':
            new_pointnumber_id = str(3) + str(AS_point)
        if Line_AS == 'SVA_South':
            new_pointnumber_id = str(4) + str(AS_point)
        id_row_fortt = {'sta': '#', 'tt': new_pointnumber_id, 'wt': None, 'phase': None}
        picked_p = pd.DataFrame()
        picked_p['sta'] = picked_p_sta
        picked_p['tt'] = np.array(picked_p_tt) - 2
        useless_array_wt = [0] * len(picked_p_sta)
        useless_array_phase = [0] * len(picked_p_sta)
        for i in range(len(useless_array_wt)):
            useless_array_wt[i] = 1
            useless_array_phase[i] = 'P'
        picked_p['wt'] = useless_array_wt
        picked_p['phase'] = useless_array_phase
        picked_p.loc[-1] = id_row_fortt
        picked_p.index = picked_p.index + 1
        picked_p = picked_p.sort_index()
        #picked_p.iloc[1:, :]
        #picked_p = picked_p.reset_index(drop=True)
        tt_PS_thisline = pd.concat([tt_PS_thisline, picked_p])
        tt_PS_thisline.to_csv(output_path + 'tt_PS' + Line_AS, index=False, sep='\t', na_rep='', header=False)      
    tt_PS_final = pd.concat([tt_PS_final, tt_PS_thisline])
    tt_PS_final.to_csv(output_path + 'tt_PS', index=False, sep='\t', na_rep='', header=False)










'''
p_files = [file_name for file_name in file_list if file_name.endswith("_P.dat")]
for p_files_phasenet in p_files:
    data = np.loadtxt('/raid1/share/for_HM_2/phasenet_result/' + p_files_phasenet)
    plt.plot([data[0], data[0]], [data[1]*10000-l_redbar, data[1]*10000+l_redbar], 'b')
    #plt.ylim(0, 2500)
plt.savefig(siesmic_profile_output + Line_AS + '_' + str(AS_point) + '_new_phasenet.png', dpi=300)
'''












'''
if ylim_dist == 'no':
    pass
else:
    plt.ylim(ylim_dist)
plt.xlim(xlim_p)
plt.rcParams['font.sans-serif'] =  'Nimbus Roman'
plt.ylabel('Distance [m]')
plt.xlabel('Time [sec]')
plt.title('Seismic Reflection Profile')

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
im = ax1.scatter(all_dist_sle, picked_p_tt, s=8, c=np.array(snr), cmap='jet')

#ax2.scatter(all_dist_sle, np.array(snr), s=6, color='green')
ax1.set_ylabel('Time [sec]')
#ax2.set_ylabel('SNR', color='green')
ax1.set_xlabel('Distance [m]')
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label('SNR')
fig.suptitle('P-Arrival Picking and SNR', fontsize=16, y=0.88, va = 'bottom')
'''

# %%

