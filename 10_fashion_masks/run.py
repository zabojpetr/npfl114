import argparse
import subprocess
import os

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help = 'File with settings')
    args = parser.parse_args()

    file = open(args.file, "r")

    os.system('kinit')


    for line in file.readlines():
        # subprocess.Popen('ulimit -t unlimited; nice -n 19 python3 mnist_competition.py ' + line, shell=True)
        # subprocess.run('ulimit -t unlimited; nice -n 19 python3 mnist_competition.py ' + line, shell=True)
        print("Started: " + line)
        os.system('kinit -R')
        os.system('aklog')
        os.system('klist')
        os.system('python3 fashion_masks.py ' + line)
        print()
        print()
