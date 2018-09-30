#!/usr/bin/env python
""" Script to launch tensorboard for all trained models """

import argparse
import glob
import yaml
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--port', default='6006')
parser.add_argument('yaml_files', nargs='*', default=glob.glob('*.yaml'))
cmd_args = parser.parse_args()
print(cmd_args)

logdirs = []
for file in cmd_args.yaml_files:
    name = file.replace('config/', '').replace('.yaml', '')
    data = yaml.load(open(file))
    checkpoint_dir = data.get('checkpoint-dir')
    if not checkpoint_dir:
        print('%s has no checkpoint-dir' % file)
        continue

    logdirs.append('%s:%s' % (name, checkpoint_dir))

print(logdirs)
args = ['tensorboard', '--logdir', ','.join(logdirs), '--port', cmd_args.port]
print('Running %s' % ' '.join(args))
subprocess.call(args)
