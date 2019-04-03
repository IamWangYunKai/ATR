import sys
import argparse
from time import sleep
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="mlp")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", default=0, type=int)
    args = parser.parse_args()

    print('policy:', args.policy)
    print('seed:', args.seed)

    while(True):
        print('policy:', args.policy)
        line = sys.stdin.readlines()
        sleep(0.001)
        sys.stdout.write(line)