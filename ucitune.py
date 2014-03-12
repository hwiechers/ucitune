#!python3

import datetime
import functools
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys

from os.path import exists, isdir, join

from docopt import docopt

ARGS_FILENAME  = '_ucitune_args.txt'
STATE_FILENAME = '_ucitune_state.txt'
MATCHLOG_DIR = '_ucitune_matchlogs'

CUTECHESS_CMD = 'C:/Users/Henri2/_Home/Utils/cutechess-cli/cutechess-cli.exe'

import signal, os

stop = False

workdir = None
targetname = None
basename = None
concurrency = None

root = None
matchnum = 0

def exitIfStopped():
    if stop:
        print('Terminating due to user input')
        savestate()
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
                  '-engine name=target cmd={0} option.Hash=128 option.OwnBook=false ' +
                  '"option.Tune Target={4}" '
                  '-engine name=base cmd={1} option.Hash=128 option.OwnBook=false ' +
                  '-each proto=uci option.Threads=1 tc=10+0.15')

def play_match(code, value):
    exitIfStopped()

    if not exists(rel(MATCHLOG_DIR)):
        os.mkdir(rel(MATCHLOG_DIR))

    print('{}: {} {} '.format(matchnum, code, value), end='')
    sys.stdout.flush()

    games = 2 * concurrency
    pgnpath = rel(join(MATCHLOG_DIR, '{}.pgn'.format(matchnum)))
    cmd = (CUTECHESS_CMD + ' ' + 
           cutechess_args.format(
               rel(targetname), rel(basename), concurrency, games,
               value, pgnpath))

    p = subprocess.Popen(args = cmd, stderr = subprocess.PIPE, stdout = subprocess.PIPE)
    stdout, stderr = p.communicate(timeout=60*60)
    stdout = stdout.decode('ascii')
    stderr = stderr.decode('ascii')

    if stop:
        print('')
    exitIfStopped()

    logpath = rel(join(MATCHLOG_DIR, '{}.txt'.format(matchnum)))
    with open(logpath, 'a+') as f:

        print('Match {} for {} {}'.format(matchnum, code, value), file=f)
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
    score = (int(groups['win']) + 0.5 * int(groups['draw'])) / games

    print('{0:+}'.format(score))

    return score

def rel(path):
    return join(workdir, path)

def findexe(name):
    for n in [name, name + '.exe']:
        if exists(rel(n)):
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


def savestate():
    print('Saving state... ', end='')

    with open(rel(STATE_FILENAME), 'w') as f:
        queue = []
        queue.append(root)
        while queue:
            node = queue.pop(0)

            if node.visits:
                print('{}: {}/{}'.format(node.name, node.score, node.visits), 
                      file=f)

            if node.left is not None:
                queue.append(node.left)

            if node.right is not None:
                queue.append(node.right)

    print('done')

def loadargs():
    if not exists(rel(ARGS_FILENAME)):
        raise TypeError('No args file')

    with open(rel(ARGS_FILENAME)) as f:
        targethash = f.readline().strip()
        basehash = f.readline().strip()
        numbits = int(f.readline().strip())
        numiters = int(f.readline().strip())
        concurrency = int(f.readline().strip())

    return targethash, basehash, numbits, numiters, concurrency

def loadstate():
    global root
    global matchnum

    if not exists(rel(STATE_FILENAME)):
        print('No saved state')
        return

    with open(rel(STATE_FILENAME), 'r') as f:
        nodemap = {}

        pattern = r'^(?P<nodename>(\.|[01]+)): (?P<score>\d+(\.\d+)?)/(?P<visits>\d+)$'
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            m = re.match(pattern, line) 
            if not m:
                print('Terminating due to invalid state file.')
                exit(1)
            groups = m.groupdict()

            nodename = groups['nodename']
            score = float(groups['score'])
            visits = int(groups['visits'])

            if nodename == '.':
                root = Node(None, 0, score=score, visits=visits)
                matchnum = visits
                nodemap['.'] = root
                continue

            parentname = nodename[:-1] or '.'
            parent = nodemap[parentname]
            lastbit = int(nodename[-1])

            node = Node(parent, lastbit, score=score, visits=visits)
            if not lastbit:
                parent.left = node
            else:
                parent.right = node

            nodemap[nodename] = node

    print('Loaded saved state')

def ucb(node, totalVisits):
    return (node.score / node.visits + 
            math.sqrt(2 * math.log(totalVisits)/node.visits))

def bitFormat(value, width):
    result = '{:b}'.format(value)
    result = (width - len(result)) * '0' + result
    return result
    
class Node(object):
    def __init__(self, parent, bit, score=0, visits=0):
        self.parent = parent

        if parent:
            self.bits = parent.bits + 1
            self.value = (parent.value << 1) + bit;
            self.name = bitFormat(self.value, self.bits)
        else:
            self.bits = 0
            self.value = 0
            self.name = '.'

        self.score = score
        self.visits = visits
        self.left = None
        self.right = None

    def hasUnvisited(self):
        return self.left is None or self.right is None

    def nextUnvisited(self):
        if not self.hasUnvisited():
            raise TypeError('Has no unvisited nodes')

        if self.left is None:
            self.left = Node(self, 0)
            return self.left
        else:
            self.right = Node(self, 1)
            return self.right

    def selectChild(self):
        if self.hasUnvisited():
            raise TypeError('Has unvisited nodes')

        def key(node):
            return ucb(node, self.visits)

        return max([self.left, self.right], key=key)

    #returns (code, value)
    #code is <name>+<random_bits>
    def getPlayout(self, bits):
        if bits < self.bits:
            raise TypeError('Node bits exceeds bits')

        bitsNeeded = bits - self.bits 
        if not bitsNeeded:
            return (bitFormat(self.value, self.bits), self.value)

        randomBits = random.getrandbits(bitsNeeded)

        return (bitFormat(self.value, self.bits) + '+' + bitFormat(randomBits, bitsNeeded), 
                (self.value << bitsNeeded) | randomBits)

    def update(self, score):
        self.score += score
        self.visits += 1
        if self.parent:
            self.parent.update(score)


def print_tree(root):
    bits = 0
    queue = []
    queue.append(root)

    while queue:
        node = queue.pop(0)
        print('{}:, value:{}, score:{}, visits:{}'.format(
              node.name, node.value, node.score, node.visits))

        if node.left is not None:
            print('left: ' + node.left.name)
            queue.append(node.left)

        if node.right is not None:
            print('right: ' + node.right.name)
            queue.append(node.right)

#tuneargs is (bits, iterations, concurrency) or None for resuming
def main(workdir_, tuneargs):
    global workdir
    global concurrency
    global targetname
    global basename
    global root
    global matchnum

    workdir = workdir_
    numbits = tuneargs[0] if tuneargs else None 
    numiters = tuneargs[1] if tuneargs else None
    concurrency = tuneargs[2] if tuneargs else None

    if not exists(workdir):
        print('<workdir> does not exist')
        return 1

    if not isdir(workdir):
        print('<workdir> is not a directory')
        return 1
    
    targetname = findexe('target')
    if not targetname:
        print('target executable not found in work dir')
        return 1

    basename = findexe('base')
    if not basename:
        print('base executable not found in work dir')
        return 1

    # Hash the base and target exes
    targethash = filehash(rel(targetname))
    basehash = filehash(rel(basename))
    root = None


    # Check if there is a args file 
    if exists(rel(ARGS_FILENAME)):
        if tuneargs:
            print("Search already started on workdir. Use the resume command " 
                  "to continue tuning.") 
            return 1

        saved_targethash, saved_basehash, numbits, numiters, concurrency = loadargs()
        if targethash != saved_targethash:
            print("Saved target hash doesn't match current one")
            return 1

        if basehash != saved_basehash:
            print("Saved base hash doesn't match current one")
            return 1

        loadstate()
    else:
        if not tuneargs:
            print("Search hasn't started on workdir. Use the new command to "
                  "start tuning")
            return 1

        with open(rel(ARGS_FILENAME), 'w') as f:
            print(targethash, file=f)
            print(basehash, file=f)
            print(numbits, file=f)
            print(numiters, file=f)
            print(concurrency, file=f)

        root = Node(None, 0)

    for _ in range(numiters):
        matchnum += 1

        node = root

        while node.bits < numbits and not node.hasUnvisited():
            node = node.selectChild()

        if node.bits < numbits:
            node = node.nextUnvisited()
        
        code, value = node.getPlayout(numbits)
        score = play_match(code, value)
        node.update(score)

    print_tree(root)

usage = (
"""ucitune

Usage:
  ucitune.py new <workdir> <bits> <iterations> [<concurrency>]
  ucitune.py resume <workdir>
""")

if __name__ == '__main__':
    arguments = docopt(usage)

    new = arguments['new']
    workdir = arguments['<workdir>']

    bits = None
    iterations=None
    concurrency = arguments.get('<concurrency>', '1')
    if new:
        bits = arguments['<bits>']
        if not re.match('^\d+$', bits) or int(bits) <= 0:
            print('<bits> must be a positive integer')
            exit(1)
        bits = int(bits)

        iterations = arguments['<iterations>']
        if not re.match('^\d+$', iterations) or int(iterations) <= 0:
            print('<iterations> must be a positive integer')
            exit(1)
        iterations = int(iterations)

        if not re.match('^\d+$', concurrency) or int(concurrency) <= 0:
            print('<concurrency> must be a positive integer')
            exit(1)
        concurrency = int(concurrency)

    exitcode = main(workdir, (bits, iterations, concurrency) if new else None)
    exit(exitcode)

#2. Save and load to file at end of run and every 10 matches
#3. Correctly display results
