from mido import Message, MidiFile, MidiTrack
from tqdm import tqdm

from pkg_resources import resource_filename

from nupic.data.file_record_stream import FileRecordStream
from nupic.engine import Network
from nupic.encoders import MultiEncoder
from nupic.algorithms.sdr_classifier_factory import SDRClassifierFactory
from nupic.algorithms.sdr_classifier import SDRClassifier

import numpy
numpy.set_printoptions(threshold=numpy.nan)
import utils
import os
import json
import midi

DATA_DIR = 'data/input/'
OUTPUT_DIR = 'data/output/'
CSV_DIR = 'data/csv/'
FILE = 'shelter'
FILE_TYPE = '.mid'
PARAMS_PATH = 'model_params/'

with open('model_params/encoded_model.json') as model:
    MODEL_PARAMS = json.load(model)

mid = MidiFile(DATA_DIR + FILE + FILE_TYPE)
state_matrix = utils.midiToNoteStateMatrix(DATA_DIR + FILE + FILE_TYPE)
one_track_notes = utils.noteStateMatrixToMidi(state_matrix)
track_len = len(one_track_notes[0])

x = numpy.array([[msg.statusmsg, msg.tick, msg.velocity, msg.pitch]
                 for msg in one_track_notes[0][:track_len-1]])

with open(CSV_DIR + FILE + '.csv', 'w') as csv:
    csv.write('event,tick,velocity,pitch\nint,int,int,int\n,,,\n')
    numpy.savetxt(csv, x, delimiter=',', fmt='%d')

# Init an HTM network
network = Network()

# Getting parameters for network regions
sensor_params = MODEL_PARAMS['Sensor']
spatial_pooler_params = MODEL_PARAMS['SpatialPooler']
temporal_memory_params = MODEL_PARAMS['TemporalMemory']
classifier_params = MODEL_PARAMS['Classifier']
encoders_params = MODEL_PARAMS['Encoders']

# Setting datasource
data_source = FileRecordStream(streamID=CSV_DIR+'shelter.csv')

# Adding regions to HTM network
network.addRegion('EventEncoder','ScalarSensor', json.dumps(encoders_params['event']))
network.addRegion('TickEncoder','ScalarSensor', json.dumps(encoders_params['tick']))
network.addRegion('VelocityEncoder','ScalarSensor', json.dumps(encoders_params['velocity']))
network.addRegion('PitchEncoder','ScalarSensor', json.dumps(encoders_params['pitch']))

network.addRegion('SpatialPooler', 'py.SPRegion', json.dumps(spatial_pooler_params))
network.addRegion('TemporalMemory', 'py.TMRegion', json.dumps(temporal_memory_params))

# network.addRegion('EventClassifier', 'py.SDRClassifierRegion', json.dumps(classifier_params))
# network.addRegion('TickClassifier', 'py.SDRClassifierRegion', json.dumps(classifier_params))
# network.addRegion('VelocityClassifier', 'py.SDRClassifierRegion', json.dumps(classifier_params))
# network.addRegion('PitchClassifier', 'py.SDRClassifierRegion', json.dumps(classifier_params))

# Creating inner classifiers for multifield prediction
event_classifier = SDRClassifier(steps=(1,2),alpha=0.001, actValueAlpha=0.3, verbosity=0)
tick_classifier = SDRClassifier(steps=(1,2),alpha=0.001, actValueAlpha=0.3, verbosity=0)
velocity_classifier = SDRClassifier(steps=(1,2),alpha=0.001, actValueAlpha=0.3, verbosity=0)
pitch_classifier = SDRClassifier(steps=(1,2),alpha=0.001, actValueAlpha=0.3, verbosity=0)

# Linking regions
network.link('EventEncoder', 'SpatialPooler', 'UniformLink', '')
network.link('TickEncoder', 'SpatialPooler', 'UniformLink', '')
network.link('VelocityEncoder', 'SpatialPooler', 'UniformLink', '')
network.link('PitchEncoder', 'SpatialPooler', 'UniformLink', '')

network.link('SpatialPooler', 'TemporalMemory', 'UniformLink', '', srcOutput='bottomUpOut', destInput='bottomUpIn')

# network.link('TemporalMemory', 'EventClassifier', 'UniformLink', '', srcOutput='bottomUpOut', destInput='bottomUpIn')
# network.link('TemporalMemory', 'TickClassifier', 'UniformLink', '', srcOutput='bottomUpOut', destInput='bottomUpIn')
# network.link('TemporalMemory', 'VelocityClassifier', 'UniformLink', '', srcOutput='bottomUpOut', destInput='bottomUpIn')
# network.link('TemporalMemory', 'PitchClassifier', 'UniformLink', '', srcOutput='bottomUpOut', destInput='bottomUpIn')

# network.link('EventEncoder', 'EventClassifier', 'UniformLink', '', srcOutput='bucketIdxOut', destInput='bucketIdxIn')
# network.link('EventEncoder', 'EventClassifier', 'UniformLink', '', srcOutput='actValueOut', destInput='actValueIn')
# network.link('EventEncoder', 'EventClassifier', 'UniformLink', '', srcOutput='categoryOut', destInput='categoryIn')

# network.link('TickEncoder', 'TickClassifier', 'UniformLink', '', srcOutput='bucketIdxOut', destInput='bucketIdxIn')
# network.link('TickEncoder', 'TickClassifier', 'UniformLink', '', srcOutput='actValueOut', destInput='actValueIn')
# network.link('TickEncoder', 'TickClassifier', 'UniformLink', '', srcOutput='categoryOut', destInput='categoryIn')

# network.link('VelocityEncoder', 'VelocityClassifier', 'UniformLink', '', srcOutput='bucketIdxOut', destInput='bucketIdxIn')
# network.link('VelocityEncoder', 'VelocityClassifier', 'UniformLink', '', srcOutput='actValueOut', destInput='actValueIn')
# network.link('VelocityEncoder', 'VelocityClassifier', 'UniformLink', '', srcOutput='categoryOut', destInput='categoryIn')

# network.link('PitchEncoder', 'PitchClassifier', 'UniformLink', '', srcOutput='bucketIdxOut', destInput='bucketIdxIn')
# network.link('PitchEncoder', 'PitchClassifier', 'UniformLink', '', srcOutput='actValueOut', destInput='actValueIn')
# network.link('PitchEncoder', 'PitchClassifier', 'UniformLink', '', srcOutput='categoryOut', destInput='categoryIn')
# Setting predicting field
# network.regions['Sensor'].setParameter('predictedField', 'event')

# Enable learning for all regions.
network.regions["SpatialPooler"].setParameter("learningMode", 1)
network.regions["TemporalMemory"].setParameter("learningMode", 1)

# network.regions["EventClassifier"].setParameter("learningMode", 1)
# network.regions["TickClassifier"].setParameter("learningMode", 1)
# network.regions["VelocityClassifier"].setParameter("learningMode", 1)
# network.regions["PitchClassifier"].setParameter("learningMode", 1)

# Enable inference for all regions.
network.regions["SpatialPooler"].setParameter("inferenceMode", 1)
network.regions["TemporalMemory"].setParameter("inferenceMode", 1)
# network.regions["EventClassifier"].setParameter("inferenceMode", 1)
# network.regions["TickClassifier"].setParameter("inferenceMode", 1)
# network.regions["VelocityClassifier"].setParameter("inferenceMode", 1)
# network.regions["PitchClassifier"].setParameter("inferenceMode", 1)

# Running the network
network.initialize()

# Main learning looop

N = 1
records_total = 0
for _ in tqdm(range(100)):
    data_source = FileRecordStream(streamID=CSV_DIR+FILE+'.csv')
    for iteration in tqdm(range(0, data_source.getDataRowCount(), N)):
        records_total += 1
        record = data_source.getNextRecord()
        event = record[0]
        tick = record[1]
        velocity = record[2]
        pitch = record[3]

        network.regions['EventEncoder'].setParameter('sensedValue', event)
        network.regions['TickEncoder'].setParameter('sensedValue', tick)
        network.regions['VelocityEncoder'].setParameter('sensedValue', velocity)
        network.regions['PitchEncoder'].setParameter('sensedValue', pitch)
        network.run(N)

        # Getting active cells of TM and bucket indicies of encoders to feed classifiers
        active_cells = numpy.array(network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]


        event_bucket = numpy.array(network.regions['EventEncoder'].getOutputData('bucket'))
        tick_bucket = numpy.array(network.regions['TickEncoder'].getOutputData('bucket'))
        velocity_bucket = numpy.array(network.regions['VelocityEncoder'].getOutputData('bucket'))
        pitch_bucket = numpy.array(network.regions['PitchEncoder'].getOutputData('bucket'))

        # Getting up classifiers result
        event_classifier_result = event_classifier.compute(
            recordNum = records_total,
            patternNZ = active_cells,
            classification = {
            'bucketIdx': event_bucket[0],
            'actValue': event
            },
            learn = True,
            infer = False
        )

        tick_classifier_result = tick_classifier.compute(
            recordNum = records_total,
            patternNZ = active_cells,
            classification = {
            'bucketIdx': tick_bucket[0],
            'actValue': tick
            },
            learn = True,
            infer = False
        )

        velocity_classifier_result = velocity_classifier.compute(
            recordNum = records_total,
            patternNZ = active_cells,
            classification = {
            'bucketIdx': velocity_bucket[0],
            'actValue': velocity
            },
            learn = True,
            infer = False
        )

        pitch_classifier_result = pitch_classifier.compute(
            recordNum = records_total,
            patternNZ = active_cells,
            classification = {
            'bucketIdx': pitch_bucket[0],
            'actValue': pitch
            },
            learn = True,
            infer = False
        )

        # print('Event: {}, Tick: {}, Velocity: {}, Pitch: {}\nrEvent: {}, rTick: {}, rVelocity: {}, rPitch: {}').format(event_classifier_result['actualValues'][1:2],
        #                                                             tick_classifier_result['actualValues'][1:2],
        #                                                             velocity_classifier_result['actualValues'][1:2],
        #                                                             pitch_classifier_result['actualValues'][1:2],
        #                                                             event,
        #                                                             tick,
        #                                                             velocity,
        #                                                             pitch)
    


# Instantiate a MIDI Pattern (contains a list of tracks)
pattern = midi.Pattern()
# Instantiate a MIDI Track (contains a list of MIDI events)
track = midi.Track()
# Append the track to the pattern
pattern.append(track)
seed = [144, 0, 120, 56]


for iters in tqdm(range(records_total, records_total+3000)):
    event = seed[0]
    tick = seed[1]
    velocity = seed[2]
    pitch = seed[3]

    network.regions['EventEncoder'].setParameter('sensedValue', event)
    network.regions['TickEncoder'].setParameter('sensedValue', tick)
    network.regions['VelocityEncoder'].setParameter('sensedValue', velocity)
    network.regions['PitchEncoder'].setParameter('sensedValue', pitch)
    network.run(N)

    # Getting active cells of TM and bucket indicies of encoders to feed classifiers
    active_cells = numpy.array(network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]

    event_bucket = numpy.array(network.regions['EventEncoder'].getOutputData('bucket'))
    tick_bucket = numpy.array(network.regions['TickEncoder'].getOutputData('bucket'))
    velocity_bucket = numpy.array(network.regions['VelocityEncoder'].getOutputData('bucket'))
    pitch_bucket = numpy.array(network.regions['PitchEncoder'].getOutputData('bucket'))

    # print active_cells.nonzero()

    # Getting up classifiers result
    event_classifier_result = event_classifier.compute(
        recordNum = iters,
        patternNZ = active_cells,
        classification = {
        'bucketIdx': event_bucket[0],
        'actValue': event
        },
        learn = False,
        infer = True
    )

    tick_classifier_result = tick_classifier.compute(
        recordNum = iters,
        patternNZ = active_cells,
        classification = {
        'bucketIdx': tick_bucket[0],
        'actValue': tick
        },
        learn = False,
        infer = True
    )

    velocity_classifier_result = velocity_classifier.compute(
        recordNum = iters,
        patternNZ = active_cells,
        classification = {
        'bucketIdx': velocity_bucket[0],
        'actValue': velocity
        },
        learn = False,
        infer = True
    )

    pitch_classifier_result = pitch_classifier.compute(
        recordNum = iters,
        patternNZ = active_cells,
        classification = {
        'bucketIdx': pitch_bucket[0],
        'actValue': pitch
        },
        learn = False,
        infer = True
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

    print('Event: {}, Tick: {}, Velocity: {}, Pitch: {}').format(event_classifier_result['actualValues'][ev],
                                                                tick_classifier_result['actualValues'][ti],
                                                                velocity_classifier_result['actualValues'][ve],
                                                                pitch_classifier_result['actualValues'][pi])


    if event_classifier_result['actualValues'][ev] == 144:
        midi_event = midi.NoteOnEvent(tick=int(tick_classifier_result['actualValues'][ti]),
                                      velocity=int(velocity_classifier_result['actualValues'][ve]),
                                      pitch=int(pitch_classifier_result['actualValues'][pi]))
    else:
        midi_event = midi.NoteOffEvent(tick=int(tick_classifier_result['actualValues'][ti]),
                                      velocity=int(velocity_classifier_result['actualValues'][ve]),
                                      pitch=int(pitch_classifier_result['actualValues'][pi]))

    track.append(midi_event)

    seed = [event_classifier_result['actualValues'][ev], tick_classifier_result['actualValues'][ti], velocity_classifier_result['actualValues'][ve], pitch_classifier_result['actualValues'][pi]]

# Add the end of track event, append it to the track
eot = midi.EndOfTrackEvent(tick=1)
track.append(eot)
# Save the pattern to disk
midi.write_midifile(FILE+FILE_TYPE, pattern)