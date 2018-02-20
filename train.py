import os
import glob
import argparse
import json
import utils
import numpy
import midi
import random 

from tqdm import tqdm
from htmusic.network_model import HTMusicModel

INPUT_DIR = './input/'
OUTPUT_DIR = './output/'
MODEL_DIR = './model/'
CHECKPOINT_EVERY = 5
NUM_STEPS = 20
MODEL_PARAMS = './encoded_model.json'

def get_arguments():
    parser = argparse.ArgumentParser(
        description='HTM music train script')

    parser.add_argument('--input_dir', type=str, default=INPUT_DIR,
                        help='The directory containing training data')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR,
                        help='The directory containing trained HTM model or model to train')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps per file. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--model_params', type=str, default=MODEL_PARAMS,
                        help='JSON file containing parameters for model')
    parser.add_argument('--overtrain', dest='overtrain', action='store_true',
                        help='Train on existing model')
    parser.set_defaults(overtrain=False)

    return parser.parse_args()


def main():
    args = get_arguments()
    num_steps = args.num_steps
    input_dir = args.input_dir
    model_dir = args.model_dir
    checkpoint_every = args.checkpoint_every
    model_params = args.model_params
    overtrain = args.overtrain

    if checkpoint_every > num_steps:
        print 'Warning: --checkpoint_every parameter greater then --num_steps,' \
              ' setting --checkpoint_every equal to --num_steps'
        checkpoint_every = num_steps

    if not os.path.exists(input_dir):
        raise Exception('Error: Invalid --input_dir, {} does not exist. Exiting.'.format(input_dir))

    if not os.path.exists(model_dir):
        raise Exception('Error: Invalid --model_dir, {} does not exist. Exiting.'.format(model_dir))

    if not os.path.isfile(model_params):
        raise Exception('Error: Invalid --model_params, {} does not exist. Exiting.'.format(model_params))

    with open(args.model_params, 'r') as mp:
        model_parameters = json.load(mp)

    model = HTMusicModel(model_parameters)

    if overtrain: 
        if not glob.glob(model_dir + '*.bin'):
            print 'Warning: No model files found. A new model will be created during training process.'
        else:
            model.load_model(model_dir)

    if not glob.glob(input_dir + '*.mid'):
        raise Exception('Error: No midi files found in {}. Exiting.'.format(input_dir))

    for index, file in enumerate(glob.glob(input_dir + '*.mid')):
        # TODO: rework midi file processing
        state_matrix = utils.midiToNoteStateMatrix(file)
        x = utils.noteStateMatrixToMidi(state_matrix)[0]
        track_len = len(x) - 1
        print '\nProcessing ' + file + '...'

        for _ in tqdm(range(num_steps)):
            for note in tqdm(range(0, track_len)):
                event = x[note].statusmsg
                tick = x[note].tick
                velocity = x[note].velocity
                pitch = x[note].pitch

                model.train(event, tick, velocity, pitch)

            if (_+1) % checkpoint_every == 0:
                print '\nSaving model'
                model.save_model(model_dir)

    print '\nSaving model'
    model.save_model(model_dir)


if __name__ == '__main__':
    main()
