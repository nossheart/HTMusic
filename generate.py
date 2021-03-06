import os
import glob
import argparse
import json

import random

from tqdm import tqdm
from htmusic.network_model import HTMusicModel

OUTPUT_DIR = './output/'
MODEL_DIR = './model/'
MODEL_PARAMS = './model_params.json'
DURATION = 250

def get_arguments():
    parser = argparse.ArgumentParser(
        description='HTM music generation example')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='The directory containing generated music')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR,
                        help='The directory containing trained HTM model or model to train')
    parser.add_argument('--model_params', type=str, default=MODEL_PARAMS,
                        help='JSON file containing parameters for model')
    parser.add_argument('--duration', type=int, default=DURATION,
                        help='Duration of MIDI composition in ticks')

    return parser.parse_args()


def main():
    args = get_arguments()
    output_dir = args.output_dir
    model_dir = args.model_dir
    model_params = args.model_params
    composition_duration = args.duration

    if not os.path.exists(output_dir):
        raise Exception('Error: Invalid --output_dir, {} does not exist. Exiting.'.format(output_dir))

    if not os.path.exists(model_dir):
        raise Exception('Error: Invalid --model_dir, {} does not exist. Exiting.'.format(model_dir))

    if not os.path.isfile(model_params):
        raise Exception('Error: Invalid --model_params, {} does not exist. Exiting.'.format(model_params))

    with open(args.model_params, 'r') as mp:
        model_parameters = json.load(mp)

    model = HTMusicModel(model_parameters)

    print '\nLoading model'
    model.load_model(model_dir)
    
    print '\nGenerating midi file...'
    random.seed()
    velocity = random.randint(0,128)
    pitch = random.randint(1,127)
    note_duration = round(random.uniform(1,3), 2)
    seed = [note_duration, velocity, pitch]

    model.generate(seed, output_dir, composition_duration)

if __name__ == '__main__':
    main()
