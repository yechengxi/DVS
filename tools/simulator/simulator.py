#!/usr/bin/env python

import argparse
import OpenEXR
import numpy as np
import os.path
import time
import cv2
import math
import sys
from os.path import join

import utils


class Event:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.ts = 0.0
        self.polarity = 0


def make_event(x, y, ts, pol):
    e = Event()
    e.x = x
    e.y = y
    e.ts = ts
    e.polarity = pol
    return e


class DvsSimulator:
    def __init__(self, initial_time, initial_values, C):
        assert(C > 0)
        self.C = C
        assert(initial_values.shape[0] > 0)
        assert(initial_values.shape[1] > 0)
        self.height = initial_values.shape[0]
        self.width = initial_values.shape[1]
        self.reference_values = initial_values.copy()
        self.It_array = initial_values.copy()
        self.t = initial_time
        
    def update(self, t_dt, It_dt_array):
        
        assert(It_dt_array.shape == self.It_array.shape)

        delta_t = t_dt-self.t
        assert(delta_t > 0)
        
        current_events = []
        for u in range(self.width):
            for v in range(self.height):
                events_for_px = []
                It = self.It_array[v,u]
                It_dt = It_dt_array[v,u]
                previous_crossing = self.reference_values[v,u]
                
                
                tol = 1e-6
                if math.fabs(It-It_dt) > tol: 
                    
                    polarity = +1 if It_dt >= It else -1
                    
                    list_crossings = []
                    all_crossings_found = False
                    cur_crossing = previous_crossing
                    while not all_crossings_found:
                        cur_crossing += polarity * self.C
                        if polarity > 0:
                            if cur_crossing > It and cur_crossing <= It_dt:
                                list_crossings.append(cur_crossing)
                            else:
                                all_crossings_found = True
                        else:
                            if cur_crossing < It and cur_crossing >= It_dt:
                                list_crossings.append(cur_crossing)
                            else:
                                all_crossings_found = True
                                
                    for crossing in list_crossings:
                        te = self.t + (crossing-It) * delta_t / (It_dt-It)
                        events_for_px.append(make_event(u,v,te,polarity>0))

                    current_events += events_for_px

                    if bool(list_crossings):
                        self.reference_values[v,u] = list_crossings[-1]
        
        self.It_array = It_dt_array.copy()
        self.t = t_dt
        
        return current_events


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder',
                        type=str,
                        required=True)
    parser.add_argument('--C',
                        type=float,
                        default=0.15,
                        required=False)
    parser.add_argument('--blur',
                        type=int,
                        default=0,
                        required=False)

    args = parser.parse_args()
    base_path = args.base_folder
    C = args.C
    blur_size = args.blur
    delta_event = 1.0 / 300.0
    
    efile = open(os.path.join(base_path, 'events.txt'), 'w')

    # Parse dataset
    dataset_dir = os.path.join(args.base_folder, 'rendered')
    times, img_paths, positions, orientations, cam = utils.parse_dataset(dataset_dir)   
    
    # Initialize DVS
    exr_img = OpenEXR.InputFile(join(dataset_dir, img_paths[0]))

    img = utils.extract_grayscale(exr_img)
    
    if blur_size > 0:
        img = cv2.GaussianBlur(img, (blur_size,blur_size), 0)
    
    init_sensor = utils.safe_log(img)
    init_time = float(times[0])
    last_pub_event_timestamp = init_time
    events = []
    
    # Init simulator
    sim = DvsSimulator(init_time, init_sensor, C)

    length = times[-1] - times[0]

    # Start simulation
    for frame_id in range(1, len(times)):
        timestamp = float(times[frame_id])

        print ("Processing time", timestamp - times[0], 'out of', length, 'seconds')

        exr_img = OpenEXR.InputFile(join(dataset_dir, img_paths[frame_id]))
        img = utils.extract_grayscale(exr_img)

        if blur_size > 0:
            img = cv2.GaussianBlur(img, (blur_size,blur_size), 0)

        # compute events for this frame
        img = utils.safe_log(img)
        current_events = sim.update(timestamp, img)
        events += current_events

        # publish events
        if timestamp - last_pub_event_timestamp > delta_event:
            events = sorted(events, key=lambda e: e.ts)
            
            for e in events:
                p = 0
                if (e.polarity > 0.5):
                    p = 1
                efile.write(str(e.ts) + ' ' + str(int(e.x)) + ' ' + str(int(e.y)) + ' ' + str(p) + '\n')
            
            events = []
            last_pub_event_timestamp = timestamp
          
    efile.close()  
