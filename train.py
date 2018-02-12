import os
import glob
import argparse
import json
import utils
import numpy
import midi

from tqdm import tqdm
from htmusic.model import HTMusicModel

INPUT_DIR = './input/'
OUTPUT_DIR = './output/'
MODEL_DIR = './model/'
CHECKPOINT_EVERY = 5
NUM_STEPS = 20
MODEL_PARAMS = './encoded_model.json'


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

                model.network.regions['EventEncoder'].setParameter(
                    'sensedValue', event)
                model.network.regions['TickEncoder'].setParameter(
                    'sensedValue', tick)
                model.network.regions['VelocityEncoder'].setParameter(
                    'sensedValue', velocity)
                model.network.regions['PitchEncoder'].setParameter(
                    'sensedValue', pitch)
                model.network.run(1)

                # Getting active cells of TM and bucket indicies of encoders to feed classifiers
                active_cells = numpy.array(
                    model.network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]
                event_bucket = numpy.array(
                    model.network.regions['EventEncoder'].getOutputData('bucket'))
                tick_bucket = numpy.array(
                    model.network.regions['TickEncoder'].getOutputData('bucket'))
                velocity_bucket = numpy.array(
                    model.network.regions['VelocityEncoder'].getOutputData('bucket'))
                pitch_bucket = numpy.array(
                    model.network.regions['PitchEncoder'].getOutputData('bucket'))

                # Getting up classifiers result
                event_classifier_result = model.event_classifier.compute(
                    recordNum=records_total,
                    patternNZ=active_cells,
                    classification={
                        'bucketIdx': event_bucket[0],
                        'actValue': event
                    },
                    learn=True,
                    infer=False
                )

                tick_classifier_result = model.tick_classifier.compute(
                    recordNum=records_total,
                    patternNZ=active_cells,
                    classification={
                        'bucketIdx': tick_bucket[0],
                        'actValue': tick
                    },
                    learn=True,
                    infer=False
                )

                velocity_classifier_result = model.velocity_classifier.compute(
                    recordNum=records_total,
                    patternNZ=active_cells,
                    classification={
                        'bucketIdx': velocity_bucket[0],
                        'actValue': velocity
                    },
                    learn=True,
                    infer=False
                )

                pitch_classifier_result = model.pitch_classifier.compute(
                    recordNum=records_total,
                    patternNZ=active_cells,
                    classification={
                        'bucketIdx': pitch_bucket[0],
                        'actValue': pitch
                    },
                    learn=True,
                    infer=False
                )

    print 'Generating midi file...'
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    seed = [144, 0, 120, 56]

    for iters in tqdm(range(records_total, records_total + 3000)):
        event = seed[0]
        tick = seed[1]
        velocity = seed[2]
        pitch = seed[3]

        model.network.regions['EventEncoder'].setParameter(
            'sensedValue', event)
        model.network.regions['TickEncoder'].setParameter(
            'sensedValue', tick)
        model.network.regions['VelocityEncoder'].setParameter(
            'sensedValue', velocity)
        model.network.regions['PitchEncoder'].setParameter(
            'sensedValue', pitch)
        model.network.run(1)

        # Getting active cells of TM and bucket indicies of encoders to feed classifiers
        active_cells = numpy.array(
            model.network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]

        event_bucket = numpy.array(
            model.network.regions['EventEncoder'].getOutputData('bucket'))
        tick_bucket = numpy.array(
            model.network.regions['TickEncoder'].getOutputData('bucket'))
        velocity_bucket = numpy.array(
            model.network.regions['VelocityEncoder'].getOutputData('bucket'))
        pitch_bucket = numpy.array(
            model.network.regions['PitchEncoder'].getOutputData('bucket'))

        # print active_cells.nonzero()

        # Getting up classifiers result
        event_classifier_result = model.event_classifier.compute(
            recordNum=iters,
            patternNZ=active_cells,
            classification={
                'bucketIdx': event_bucket[0],
                'actValue': event
            },
            learn=False,
            infer=True
        )

        tick_classifier_result = model.tick_classifier.compute(
            recordNum=iters,
            patternNZ=active_cells,
            classification={
                'bucketIdx': tick_bucket[0],
                'actValue': tick
            },
            learn=False,
            infer=True
        )

        velocity_classifier_result = model.velocity_classifier.compute(
            recordNum=iters,
            patternNZ=active_cells,
            classification={
                'bucketIdx': velocity_bucket[0],
                'actValue': velocity
            },
            learn=False,
            infer=True
        )

        pitch_classifier_result = model.pitch_classifier.compute(
            recordNum=iters,
            patternNZ=active_cells,
            classification={
                'bucketIdx': pitch_bucket[0],
                'actValue': pitch
            },
            learn=False,
            infer=True
        )

        ev = event_classifier_result[1].argmax()
        ti = tick_classifier_result[1].argmax()
        ve = velocity_classifier_result[1].argmax()
        pi = pitch_classifier_result[1].argmax()

        # print tick_classifier_result
        # print ev, ti, ve, pi

        # print event_classifier_result['actualValues'][ev]
        # print tick_classifier_result['actualValues'][ti]
        # print velocity_classifier_result['actualValues'][ve]
        # print pitch_classifier_result['actualValues'][pi]

        # print('Event: {}, Tick: {}, Velocity: {}, Pitch: {}').format(event_classifier_result['actualValues'][ev],
        #                                                             tick_classifier_result['actualValues'][ti],
        #                                                             velocity_classifier_result['actualValues'][ve],
        #                                                             pitch_classifier_result['actualValues'][pi])

        if event_classifier_result['actualValues'][ev] == 144:
            midi_event = midi.NoteOnEvent(tick=int(tick_classifier_result['actualValues'][ti]),
                                          velocity=int(
                                              velocity_classifier_result['actualValues'][ve]),
                                          pitch=int(pitch_classifier_result['actualValues'][pi]))
        else:
            midi_event = midi.NoteOffEvent(tick=int(tick_classifier_result['actualValues'][ti]),
                                           velocity=int(
                                               velocity_classifier_result['actualValues'][ve]),
                                           pitch=int(pitch_classifier_result['actualValues'][pi]))

        track.append(midi_event)

        seed = [event_classifier_result['actualValues'][ev], tick_classifier_result['actualValues'][ti],
                velocity_classifier_result['actualValues'][ve], pitch_classifier_result['actualValues'][pi]]

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Save the pattern to disk
    midi.write_midifile(output_dir + 'example.mid', pattern)


if __name__ == '__main__':
    main()
