import datetime
import random
import os
import pandas as pd
import default_process

def bp_event_lists(dir_events='event_list/'):
    filenames = ['Events_Dawn_Active.txt']
    #filenames = ['Events_Dawn_Active.txt','Events_Dawn_Moderate.txt','Events_Dawn_Quiet.txt',
    #             'Events_Dusk_Active.txt','Events_Dusk_Moderate.txt','Events_Dusk_Quiet.txt']
    #filenames = ['Events_Dawn_Active.txt','Events_Dawn_Moderate.txt','Events_Dawn_Quiet.txt']
    #filenames = ['Events_Dusk_Active.txt','Events_Dusk_Moderate.txt','Events_Dusk_Quiet.txt']
    stretchMethod = 'wavelets'
    process_method = 'equal_loudness'

    for filename in filenames:
        df = pd.read_csv(dir_events+filename,header=0,
                         delim_whitespace=True)
        StartTime = df["StartTime"]
        st_dt = [datetime.datetime.strptime(ii, '%Y-%m-%d/%H:%M:%S') for ii in StartTime]
        StopTime = df["StopTime"]
        et_dt = [datetime.datetime.strptime(ii, '%Y-%m-%d/%H:%M:%S') for ii in StopTime]
        #randomly select 2 events to plot
        #rd_ind = random.sample(range(0, len(st_dt)), 2)
        #rd_ind=[0,1] #select the first two events for testing
        
        #plot all events
        rd_ind=range(len(st_dt))
        filename_str=filename[7:12]+stretchMethod
        probe='the'
        
        for ii in rd_ind:
            start_time = st_dt[ii]
            print('Event start time: %s' %str(start_time))
            end_time = et_dt[ii]
            print('Event end time: %s' %str(end_time))
            #print(df.iloc[[ii]])
          
            directory = 'outputs/'+filename_str+'_'+start_time.strftime("%Y%m%d")
            if not os.path.exists(directory):
                os.mkdir(directory)
            fn = probe.upper()+'_orbit_info_'+start_time.strftime("%Y%m%d")+'_'+end_time.strftime("%Y%m%d")+'.txt'
            df.iloc[[ii]].to_csv(directory+'/'+fn, index=False, header=True)
            print('Write to orbit info file finished!')
            
            default_process.process_data(start_time=start_time, end_time=end_time,probe=probe, 
                                         filetype = ['wav','ogg'],filename_str=filename_str,
                                         stretchMethod=stretchMethod,process_method=process_method)
bp_event_lists()
