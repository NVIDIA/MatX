#!/usr/bin/python3

"""
Auto unit test vector generator. Looks in the GENERATORS_DIR directory for any executable scripts or binaries, and
runs them to generate test vectors. Writes out a manifest file for caching vectors that have already been 
generated.
"""

import os
import subprocess
import hashlib
import csv
import pathlib

GENERATORS_DIR_REL = "generators/"
GENERATED_DIR  = "generated/"
MANIFEST_FILE_REL = "manifest.txt"

cur_path = str(pathlib.Path(__file__).parent.absolute())
cwd = os.getcwd()

abs_matx = cur_path[:cur_path.find('/matx/') + len('/matx/')]
GENERATORS_DIR = f"{cur_path}/{GENERATORS_DIR_REL}"
MANIFEST_FILE = f"{cwd}/{MANIFEST_FILE_REL}"
print("Running test vector pre-flight check script", flush=True)

manifest = {}


if not os.path.isdir(GENERATED_DIR):
    os.mkdir(GENERATED_DIR)

try:
    with open(MANIFEST_FILE) as ml:
        lines = ml.readlines()
        for line in lines:
            line = line.split(',')
            manifest[line[0].strip()] = line[1].strip()
except FileNotFoundError:
    print('No test vectors generated yet. Regenerating all...', flush=True)

for _, _, files in os.walk(GENERATORS_DIR, topdown=False):
    for f in files:
        if f[-3:] != '.py' or f == 'matx_common.py':
            continue
        hash = hashlib.md5(open(GENERATORS_DIR + f,'rb').read()).hexdigest().strip()
        if f not in manifest or manifest[f] != hash:
            print(f"Regenerating {f}", flush=True)
            try:
                p = subprocess.check_output(GENERATORS_DIR + f, cwd=GENERATED_DIR)
                manifest[f] = hash
            except subprocess.CalledProcessError as ec:
                print(f"Calling script {f} failed with error code {ec.returncode}: {ec.output}", flush=True)

m = open(MANIFEST_FILE, "w")
for k, v in manifest.items():
    m.write(f"{k},{v}\n")