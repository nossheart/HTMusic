import os
import glob
import argparse
import json
import utils
import numpy
import midi

from tqdm import tqdm
from htmusic.api_model import HTMusicModel

INPUT_DIR = './input/'
OUTPUT_DIR = './output/'
MODEL_DIR = './model/'
CHECKPOINT_EVERY = 5
NUM_STEPS = 20
MODEL_PARAMS = './model_params.json'


def get_arguments():
    parser = argparse.ArgumentParser(
        description='HTM music generation example')

    parser.add_argument('--input_dir', type=str, default=INPUT_DIR,
                        help='The directory containing training data')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='The directory containing generated music')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR,
                        help='The directory containing trained HTM model or model to train')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps per file. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--model_params', type=str, default=MODEL_PARAMS,
                        help='JSON file containing parameters for model')

    return parser.parse_args()


def main():
    args = get_arguments()
    num_steps = args.num_steps
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    checkpoint_every = args.checkpoint_every

    with open(args.model_params, 'r') as mp:
        model_parameters = json.load(mp)

    model = HTMusicModel(model_parameters)

    records_total = 0

    for file in glob.glob(input_dir + '*.mid'):

        # TODO: rework midi file processing
        state_matrix = utils.midiToNoteStateMatrix(file)
        x = utils.noteStateMatrixToMidi(state_matrix)[0]
        track_len = len(x) - 1
        print 'Processing ' + file + ' ...'

        for _ in tqdm(range(num_steps)):
            for note in tqdm(range(0, track_len)):
                records_total += 1
                event = x[note].statusmsg
                tick = x[note].tick
                velocity = x[note].velocity
                pitch = x[note].pitch

                model.train(event, tick, velocity, pitch, records_total)

    print 'Generating midi file...'
    seed = [144, 0, 40, 56]
    model.generate(seed, records_total, output_dir)

if __name__ == '__main__':
    main()
