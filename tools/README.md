dataset tools
=============

Tools and helper scripts to facilitate data conversion from raw text or ROS .bag files to .npy/.npz format.

## BRIEF MANUAL:
1. Go to 'pydvs' folder and run *sudo ./setup.py install* - this will install some C tools and make them accessible from Python. Everything here is described for Python 3

2. To convert the MVSEC .bag file dataset to the text format, run:

```
cd mvsec/tools/gt_flow
./run_test.py --mvsec_dir ~/storage/test/ --ename outdoor_night --eid 1 --mode 1
```

The output path is hardcoded to be */home/ncos/Desktop/MVSEC/* in the *compute_flow.py* (sorry for that). Also, all output directories have to exist...

3. To convert text format to .npz, run the *dataset_gen.py*, e.g:
```
./dataset_gen.py --base_dir ~/Desktop/MVSEC/outdoor_night_1
```

4. To work with the simulated data, cd into *simulator* directory
  - blender scene.blend -b --python blenderscript.py -a # Will render the scene and save rgb/depth/trajectories and object masks in the *rendered* directory
  - *simulator.py* can create the text event file
  - *dataset_gen.py* will convert the text format + object masks and trajectories into the .npz file
