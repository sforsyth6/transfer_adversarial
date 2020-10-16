import re
import argparse

parser = argparse.ArgumentParser(description='Experiment')

parser.add_argument('--file')
parser.add_argument('--line')
parser.add_argument('--linesper')

args = parser.parse_args()

file_contents = open(args.file).readlines()

line = int(args.line)
p = int(args.linesper)

for i in range(0,45):
    t = file_contents[i *  p + line].rstrip().split(":")
    if i == 0:
        print(t[0].strip())
    if len(t) > 1:
        print(t[1].strip())
