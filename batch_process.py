import datetime
import random
import pandas as pd
import default_process

def bp_event_lists():
    #filenames = ['Events_Dawn_Active.txt','Events_Dawn_Moderate.txt','Events_Dawn_Quiet.txt']
    filenames = ['Events_Dusk_Active.txt','Events_Dusk_Moderate.txt','Events_Dusk_Quiet.txt']
    for filename in filenames:
        df = pd.read_csv('event_list/'+filename,header=0,
                         delim_whitespace=True)
        StartTime = df["StartTime"]
        st_dt = [datetime.datetime.strptime(ii, '%Y-%m-%d/%H:%M:%S') for ii in StartTime]
        StopTime = df["StopTime"]
        et_dt = [datetime.datetime.strptime(ii, '%Y-%m-%d/%H:%M:%S') for ii in StopTime]
        #randomly select 2 events to plot
        rd_ind = random.sample(range(0, len(st_dt)), 2)
        for ii in rd_ind:
            start_time = st_dt[ii]
            print('Event start time: %s' %str(start_time))
            end_time = et_dt[ii]
            print('Event end time: %s' %str(end_time))
            default_process.process_data(start_time=start_time, end_time=end_time,probe='the', 
                                         spacing = 3., pos_min = 5, stretch = 6, samplerate = 44100, 
                                         filetype = ['wav','ogg'],filename_str=filename.strip(".txt"))
bp_event_lists()