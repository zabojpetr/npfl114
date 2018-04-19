import argparse
import subprocess
import os

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="test.txt", type=str, help = 'File with settings')
    args = parser.parse_args()

    file = open(args.file, "r")

    for line in file.readlines():
        # subprocess.Popen('ulimit -t unlimited; nice -n 19 python3 mnist_competition.py ' + line, shell=True)
        # subprocess.run('ulimit -t unlimited; nice -n 19 python3 mnist_competition.py ' + line, shell=True)
        print("Started: " + line)
        os.system('python sequence_prediction.py ' + line)
        print()
        print()
