#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract dV from NAMD output """

import re
import os
import sys
import fnmatch
import argparse
import itertools

__author__ = "Jérôme Eberhardt, Roland H Stote, and Annick Dejaegere"
__copyright__ = "Copyright 2016, Jérôme Eberhardt"
__credits__ = ["Jérôme Eberhardt", "Roland H Stote", "Annick Dejaegere"]

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"

def getFname(dir, s):
    return dir + os.sep + s

def get_files(dir, pattern):
    
    files = fnmatch.filter(os.listdir(dir), pattern)
    files = [os.path.abspath(getFname(dir, sn)) for sn in files]
    #files = [getFname(dir, sn) for sn in files]
    RE_DIGIT = re.compile(r'(\d+)')
    ALPHANUM_KEY = lambda s: [int(g) if g.isdigit() else g \
                              for g in RE_DIGIT.split(s)]
    files.sort(key = ALPHANUM_KEY)
    
    return files

def extract_dV_from_namd_output(namd_output, interval):

    data = []

    with open(namd_output) as f:
        for line in f:
            if re.search('^ACCELERATED MD', line):
                splited_line = line.split(' ')

                time = int(splited_line[3])

                if time % interval == 0 and time != 0:
                    data.append([time, float(splited_line[5])])

    return data

def write_data(data, output_name):
    with open(output_name, 'w') as w:
        old_time = None
        i = 0

        for d in data:
            #On enleve toutes duplications
            if d[0] != old_time:
                w.write("%10d %5.3f\n" % (d[0], d[1]))
                i += 1

            old_time = d[0]

    print "Number of line: %s" % i

def cmdlineparse():
    parser = argparse.ArgumentParser(description="extract dV from NAMD output")
    parser.add_argument('-d', "--dir", dest="directories", required=True, \
                        action="store", type=str, nargs='+', \
                        help='list of output directories')
    parser.add_argument("-i", "--interval", dest="interval", required=False, \
                        action="store", default=1, type=int,
                        help="interval we take the dV")
    parser.add_argument("-o", "--output", dest="output_name", required=False, \
                        action="store", default='weights.dat', type=str,
                        help="name of the output file with weights")

    args = parser.parse_args()

    return args

def main():

    args = cmdlineparse()

    data = []

    for directory in args.directories:
        output_namd_files = get_files(directory, '*-prod*.out')

        for output_namd_file in output_namd_files:
            data += extract_dV_from_namd_output(output_namd_file, args.interval)

    write_data(data, args.output_name)

if __name__ == '__main__':
    main()
    sys.exit(0)