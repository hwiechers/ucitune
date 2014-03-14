#!python3

import datetime
import functools
import hashlib
import json
import numpy
import os
import random
import re
import signal
import subprocess
import sys

from os.path import exists, isdir, join

from collections import namedtuple
from docopt import docopt
from math import exp, sqrt
from scipy.stats import norm
from numpy.linalg import inv
from numpy import array, identity, matrix

ARGS_FILENAME  = '_ucitune_args.txt'
STATE_FILENAME = '_ucitune_state.txt'
MATCHLOG_DIR = '_ucitune_matchlogs'

GAMES_PER_MATCH = 2 #100
SAMPLE_VARIANCE = (0.5 ** 2) / GAMES_PER_MATCH #Assumed
ASPIRATION_ELO = 2

import signal, os

stop = False

aspiration_score = 1 / (1 + 10 ** (-ASPIRATION_ELO / 400)) - 0.5

Sample = namedtuple('Sample', ['value', 'number', 'mean', 'variance'])

def exitIfStopped():
    if stop:
        print('Terminating due to user input')
        exit(1)

def handleSIGINT(signum, frame):
    global stop
    if not stop:
        stop = True
    else:
        # Forget about safe exit and just leave
        exit(1)

signal.signal(signal.SIGINT, handleSIGINT)

cutechess_args = ('-repeat -rounds {3} -tournament gauntlet -pgnout {5} ' + 
                  '-resign movecount=3 score=400 ' + 
                  '-draw movenumber=34 movecount=8 score=20 ' +
                  '-concurrency {2} ' + 
                  '-openings file=8moves_v3.pgn format=pgn order=random plies=16 ' + 
                  '-engine name=target cmd={0} option.Hash=128 ' +
                  '"option.Tune Target={4}" '
                  '-engine name=base cmd={1} option.Hash=128 ' +
                  '-each proto=uci option.Threads=1 tc=10+0.15')

def play_match(workdir, targetname, basename, 
            value, cutechess_path, concurrency, samplenum):
    exitIfStopped()

    if not exists(join(workdir, MATCHLOG_DIR)):
        os.mkdir(join(workdir, MATCHLOG_DIR))

    pgnpath = join(workdir, join(MATCHLOG_DIR, '{}.pgn'.format(samplenum)))
    cmd = (cutechess_path + ' ' + 
           cutechess_args.format(
               join(workdir, targetname), join(workdir, targetname), concurrency, GAMES_PER_MATCH,
               value, pgnpath))

    p = subprocess.Popen(args = cmd, stderr = subprocess.PIPE, stdout = subprocess.PIPE)
    stdout, stderr = p.communicate(timeout=60*60)
    stdout = stdout.decode('ascii')
    stderr = stderr.decode('ascii')

    if stop:
        print('')
    exitIfStopped()

    logpath = join(workdir, join(MATCHLOG_DIR, '{}.txt'.format(samplenum)))
    with open(logpath, 'a+') as f:

        print('Sample {} for {}'.format(samplenum, value), file=f)
        print(str(datetime.datetime.now()), file=f)

        print('cmd:', file=f)
        print(cmd, file=f)

        print('stdout:', file=f)
        print(stdout, file=f)

        print('stderr:', file=f)
        print(stderr, file=f)

    if stderr or 'Warning' in stdout:
        print('Terminating due to cutechess error/warning')
        exit(1)

    pattern = ('Score of target vs base: ' + 
               '(?P<win>\d+) - (?P<loss>\d+) - (?P<draw>\d+)')
    groups = [m.groupdict() for m in re.finditer(pattern, stdout)][-1]
    score = (int(groups['win']) + 0.5 * int(groups['draw'])) / GAMES_PER_MATCH - 0.5

    return score

def findexe(workdir, name):
    for n in [name, name + '.exe']:
        if exists(join(workdir, n)):
            return n
    else:
        return None


# Taken from http://bugs.python.org/issue17436
def filehash(path):
    blocksize = 64*1024
    sha = hashlib.sha256()
    with open(path, 'rb') as fp:
        while True:
            data = fp.read(blocksize)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()

def saveargs(workdir, targethash, param_base_val, param_min, param_max):
    with open(join(workdir, ARGS_FILENAME), 'w') as f:
        print(targethash, file=f)
        print(param_base_val, file=f)
        print(param_min, file=f)
        print(param_max, file=f)

def loadargs(workdir):
    if not exists(join(workdir, ARGS_FILENAME)):
        raise TypeError('No args file')

    with open(join(workdir, ARGS_FILENAME)) as f:
        targethash = f.readline().strip()
        param_base_val = int(f.readline().strip())
        param_min = int(f.readline().strip())
        param_max = int(f.readline().strip())

    return targethash, param_base_val, param_min, param_max

def loadstate():
    if not exists(join(workdir, STATE_FILENAME)):
        print('No saved state')
        return

    matchnum = None
    samples = []
    with open(join(workdir, STATE_FILENAME), 'r') as f:
        matchnum = int(f.readline())

        for line in f:
            components = line.split(',')
            samples.append(Sample(
                    int(components[0]), 
                    int(components[1]), 
                    float(components[2]), 
                    float(components[3])))

    print('Loaded saved state')

    return matchnum, samples

def savestate(matchnum, samples):

    with open(join(workdir, STATE_FILENAME), 'w') as f:
        f.write('{}\n'.format(matchnum))
        for sample in samples:
            f.write('{},{},{},{}\n'.format( 
                    sample.value, sample.number,
                    sample.mean, sample.variance))



def kernel(x, y):
    return 0.25 * exp( -((x - y)**2) / (2 * 25**2) )

def main(workdir, resume, param_base_val=None, param_min=None, param_max=None):

    configpath = get_configpath()
    if not exists(configpath):
        print('Use the config command first')
        return 1

    if isdir(configpath):
        print('The config path is a directory')
        return 1

    with openconfig('r') as f:
        lines = f.readlines()
        if len(lines) < 2: 
            print('Invalid config')
            return 1
        cutechess_path = lines[0].strip()
        concurrency = lines[1].strip()

    if not exists(workdir):
        print('<workdir> does not exist')
        return 1

    if not isdir(workdir):
        print('<workdir> is not a directory')
        return 1
    
    targetname = findexe(workdir, 'target')
    if not targetname:
        print('target executable not found in work dir')
        return 1

    basename = findexe(workdir, 'base')
    if not basename:
        print('base executable not found in work dir')
        return 1

    # Hash the base and target exes
    targethash = filehash(join(workdir, targetname))

    # Check if there is a args file 
    if exists(join(workdir, ARGS_FILENAME)):
        if not resume:
            print("Search already started on workdir. Use the 'resume' " 
                  "command to continue tuning.") 
            return 1

        saved_targethash, param_base_val, param_min, param_max = loadargs(workdir)
        if targethash != saved_targethash:
            print("Saved target hash doesn't match current one")
            return 1

        matchnum, samples = loadstate()
    else:
        if resume:
            print("Search hasn't started on workdir. Use the 'new' command to "
                  "start tuning")
            return 1

        saveargs(workdir, targethash, param_base_val, param_min, param_max)

        matchnum = 0
        samples = [Sample(param_base_val, 1, 0, 0)]

        savestate(matchnum, samples)

    def getvariance(value):
        sample = [x for x in samples if x.value == value]
        if sample:
            return sample[0].variance
        else:
            return SAMPLE_VARIANCE

    while True:

        size = len(samples)

        covariance = numpy.matrix(
                [[(kernel(r[0], c[0]) + 
                   (getvariance(r[0]) if r[0] == c[0] else 0))
                    for c in samples] 
                    for r in samples],
                numpy.double)
        covinv = inv(covariance)

        bestnode = param_min
        bestcost = 1
        for x in range(param_min, param_max + 1):
            if x == param_base_val:
                continue

            covvec = numpy.matrix(
                    [kernel(x, y[0]) for y in samples], 
                    numpy.double).reshape((size, 1))

            resvec = numpy.matrix(
                    [y.mean for y in samples],
                    numpy.double).reshape((size, 1))

            mean = (covvec.T * covinv * resvec).item(0)
            variance = (kernel(0, 0) + SAMPLE_VARIANCE - covvec.T * covinv * covvec).item(0)
            cost = norm.cdf((aspiration_score - mean) / sqrt(variance))

            if cost < bestcost:
                bestnode, bestcost = x, cost

        value = bestnode
        matchnum += 1

        print('{}: {} {:.4f} '.format(matchnum, value, 1 - cost, end=''))
        sys.stdout.flush()

        score = play_match(workdir, targetname, basename, 
                value, cutechess_path, concurrency, matchnum)
        print(score)

        oldSample = [x for x in samples if x.value == value] 
        if oldSample:
            sample = oldSample[0]
            samples.remove(sample)
            samples.append(Sample(
                value, 
                sample.number + 1,
                (sample.number * sample.mean + score) / (sample.number + 1),
                (sample.number ** 2 * sample.variance + SAMPLE_VARIANCE) / 
                (sample.number + 1)**2))
        else: 
            samples.append(Sample(value, 1, score, SAMPLE_VARIANCE))
        savestate(matchnum, samples)

#TODO: Figure out a good temperature schedule
def simulated_annealing(costfn, neighborfn, node):
    cost = costfn(node)
    for temp in range(1, 0, -1):
        neighbors = neighborfn(node)
        neighbor = random.choice(neighbors)
        neighbor_cost = costfn(neighbor)
        if (neighbor_cost < cost): #or random.uniform(0, 1) < exp((cost - neighbor_cost) / temp)):
            node, cost = neighbor, neighbor_cost
    return node, cost

def get_configpath():
    return join(os.path.dirname(__file__), 'config.txt')

def openconfig(mode):
    configpath = get_configpath()
    return open(configpath, mode)

def saveconfig(cutechesscmd, concurrency):
    with openconfig('w+') as f:
        print(cutechesscmd, file=f)
        print(concurrency, file=f)

usage = (
"""ucitune

Usage:
  ucitune.py config <cutechesscmd> <concurrency>
  ucitune.py new <workdir> <base_val> <min> <max>
  ucitune.py resume <workdir>
""")

if __name__ == '__main__':
    arguments = docopt(usage)

    if arguments['config']:
        #TODO: Validate args
        saveconfig(arguments['<cutechesscmd>'], arguments['<concurrency>'])
        exit(0)

    new = arguments['new']
    resume = arguments['resume']
    workdir = arguments['<workdir>']

    param_base_val = None
    param_min = None
    param_max = None
    if new:
        #TODO: Validate args
        param_base_val = int(arguments['<base_val>'])
        param_min = int(arguments['<min>'])
        param_max = int(arguments['<max>'])

    exitcode = main(workdir, resume, param_base_val, param_min, param_max)
    exit(exitcode)
