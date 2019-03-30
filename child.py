import time
import argparse
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="mlp")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", default=0, type=int)
    args = parser.parse_args()

    time.sleep(2)
    print('policy:', args.policy)
    time.sleep(1)
    print('seed:', args.seed)