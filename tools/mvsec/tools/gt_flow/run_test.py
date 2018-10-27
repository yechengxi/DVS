#!/usr/bin/python

import compute_flow

import downloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mvsec_dir',
                    type=str,
                    help="Path to MVSEC directory.",
                    required=True)

parser.add_argument('--ename',
                    type=str,
                    help="Experiment name.",
                    required=True)

parser.add_argument('--eid',
                    type=int,
                    help="Experiment number.",
                    required=True)

parser.add_argument('--mode',
                    type=int,
                    help="Mode of operation.",
                    required=True)

args = parser.parse_args()

downloader.set_tmp(args.mvsec_dir)

print zip(downloader.experiments, downloader.number_of_runs)


compute_flow.experiment_flow(args.ename, args.eid, mode=args.mode)


exit(0)
for experiment, n_runs in zip(downloader.experiments, downloader.number_of_runs):
    for i in range(n_runs):
        run_number = i+1
        print "Running ", experiment, run_number
        compute_flow.experiment_flow(experiment, run_number, mode=args.mode)
        exit(0)
