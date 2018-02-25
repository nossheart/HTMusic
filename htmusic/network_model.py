import capnp
from nupic.engine import Network
from nupic.proto.SdrClassifier_capnp import SdrClassifierProto
from nupic.proto.SpatialPoolerProto_capnp import SpatialPoolerProto
from nupic.algorithms.sdr_classifier import SDRClassifier
from tqdm import tqdm

import pretty_midi
import random
import json
import numpy
import os
import datetime


class HTMusicModel(object):

    def __init__(self, model_params):
        # Init an HTM network
        self.network = Network()

        # Getting parameters for network regions
        self.sensor_params = model_params['Sensor']
        self.spatial_pooler_params = model_params['SpatialPooler']
        self.temporal_memory_params = model_params['TemporalMemory']
        self.classifiers_params = model_params['Classifiers']
        self.encoders_params = model_params['Encoders']

        # Adding regions to HTM network
        self.network.addRegion('DurationEncoder', 'ScalarSensor',
                               json.dumps(self.encoders_params['duration']))
        self.network.addRegion('VelocityEncoder', 'ScalarSensor', json.dumps(
            self.encoders_params['pitch']))
        self.network.addRegion('PitchEncoder', 'ScalarSensor', json.dumps(
            self.encoders_params['velocity']))

        self.network.addRegion(
            'SpatialPooler', 'py.SPRegion', json.dumps(self.spatial_pooler_params))
        self.network.addRegion(
            'TemporalMemory', 'py.TMRegion', json.dumps(self.temporal_memory_params))

        # Creating outer classifiers for multifield prediction
        dclp = self.classifiers_params['duration']
        vclp = self.classifiers_params['pitch']
        pclp = self.classifiers_params['velocity']

        self.duration_classifier = SDRClassifier(steps=(1,), verbosity=dclp['verbosity'], alpha=dclp['alpha'],
                                                 actValueAlpha=dclp['actValueAlpha'])
        self.velocity_classifier = SDRClassifier(steps=(1,), verbosity=vclp['verbosity'], alpha=vclp['alpha'],
                                                 actValueAlpha=vclp['actValueAlpha'])
        self.pitch_classifier = SDRClassifier(steps=(1,), verbosity=pclp['verbosity'], alpha=pclp['alpha'],
                                              actValueAlpha=pclp['actValueAlpha'])

        self._link_all_regions()
        self._enable_learning()
        self._enable_inference()

        self.network.initialize()

    def _link_all_regions(self):
        # Linking regions
        self.network.link('DurationEncoder',
                          'SpatialPooler', 'UniformLink', '')
        self.network.link('VelocityEncoder',
                          'SpatialPooler', 'UniformLink', '')
        self.network.link('PitchEncoder', 'SpatialPooler', 'UniformLink', '')
        self.network.link('SpatialPooler', 'TemporalMemory', 'UniformLink',
                          '', srcOutput='bottomUpOut', destInput='bottomUpIn')

    def _enable_learning(self):
        # Enable learning for all regions.
        self.network.regions["SpatialPooler"].setParameter("learningMode", 1)
        self.network.regions["TemporalMemory"].setParameter("learningMode", 1)

    def _enable_inference(self):
        # Enable inference for all regions.
        self.network.regions["SpatialPooler"].setParameter("inferenceMode", 1)
        self.network.regions["TemporalMemory"].setParameter("inferenceMode", 1)

    def train(self, duration, pitch, velocity):
        records_total = self.network.regions['SpatialPooler'].getSelf(
        ).getAlgorithmInstance().getIterationNum()

        self.network.regions['DurationEncoder'].setParameter(
            'sensedValue', duration)
        self.network.regions['PitchEncoder'].setParameter(
            'sensedValue', pitch)
        self.network.regions['VelocityEncoder'].setParameter(
            'sensedValue', velocity)
        self.network.run(1)

        # Getting active cells of TM and bucket indicies of encoders to feed classifiers
        active_cells = numpy.array(
            self.network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]
        duration_bucket = numpy.array(
            self.network.regions['DurationEncoder'].getOutputData('bucket'))
        pitch_bucket = numpy.array(
            self.network.regions['PitchEncoder'].getOutputData('bucket'))
        velocity_bucket = numpy.array(
            self.network.regions['VelocityEncoder'].getOutputData('bucket'))

        duration_classifier_result = self.duration_classifier.compute(
            recordNum=records_total,
            patternNZ=active_cells,
            classification={
                'bucketIdx': duration_bucket[0],
                'actValue': duration
            },
            learn=True,
            infer=False
        )

        pitch_classifier_result = self.pitch_classifier.compute(
            recordNum=records_total,
            patternNZ=active_cells,
            classification={
                'bucketIdx': pitch_bucket[0],
                'actValue': pitch
            },
            learn=True,
            infer=False
        )

        velocity_classifier_result = self.velocity_classifier.compute(
            recordNum=records_total,
            patternNZ=active_cells,
            classification={
                'bucketIdx': velocity_bucket[0],
                'actValue': velocity
            },
            learn=True,
            infer=False
        )

    def generate(self, seed, output_dir, event_amount):
        records_total = self.network.regions['SpatialPooler'].getSelf().getAlgorithmInstance().getIterationNum()

        seed = seed

        midi = pretty_midi.PrettyMIDI()
        midi_program = pretty_midi.instrument_name_to_program(
            'Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=midi_program)
        clock = 0
        for iters in tqdm(range(records_total, records_total + event_amount)):
            duration = seed[0]
            pitch = seed[1]
            velocity = seed[2]

            self.network.regions['DurationEncoder'].setParameter(
                'sensedValue', duration)
            self.network.regions['PitchEncoder'].setParameter(
                'sensedValue', pitch)
            self.network.regions['VelocityEncoder'].setParameter(
                'sensedValue', velocity)
            self.network.run(1)

            # Getting active cells of TM and bucket indicies of encoders to feed classifiers
            active_cells = numpy.array(
                self.network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]

            duration_bucket = numpy.array(
                self.network.regions['DurationEncoder'].getOutputData('bucket'))

            pitch_bucket = numpy.array(
                self.network.regions['PitchEncoder'].getOutputData('bucket'))

            velocity_bucket = numpy.array(
                self.network.regions['VelocityEncoder'].getOutputData('bucket'))

            # Getting up classifiers result

            duration_classifier_result = self.duration_classifier.compute(
                recordNum=records_total,
                patternNZ=active_cells,
                classification={
                    'bucketIdx': duration_bucket[0],
                    'actValue': duration
                },
                learn=False,
                infer=True
            )

            pitch_classifier_result = self.pitch_classifier.compute(
                recordNum=records_total,
                patternNZ=active_cells,
                classification={
                    'bucketIdx': pitch_bucket[0],
                    'actValue': pitch
                },
                learn=False,
                infer=True
            )

            velocity_classifier_result = self.velocity_classifier.compute(
                recordNum=records_total,
                patternNZ=active_cells,
                classification={
                    'bucketIdx': velocity_bucket[0],
                    'actValue': velocity
                },
                learn=False,
                infer=True
            )

            du = duration_classifier_result[1].argmax()
            pi = pitch_classifier_result[1].argmax()
            ve = velocity_classifier_result[1].argmax()

            duration_top_probs = duration_classifier_result[1][0:2] / duration_classifier_result[1][0:2].sum()

            predicted_duration = duration_classifier_result['actualValues'][du]

            # predicted_duration = duration_classifier_result['actualValues'][du]
            predicted_pitch = pitch_classifier_result['actualValues'][pi]
            predicted_velocity = velocity_classifier_result['actualValues'][ve]

            # print duration_classifier_result

            note = pretty_midi.Note(velocity=int(predicted_velocity),
                                    pitch=int(predicted_pitch),
                                    start=float(clock),
                                    end=float(clock+predicted_duration))

            piano.notes.append(note)

            clock = clock + 0.25

            seed[0] = predicted_duration
            seed[1] = predicted_pitch
            seed[2] = predicted_velocity

        midi.instruments.append(piano)
        midi.remove_invalid_notes()
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%m:%S')
        midi.write(output_dir + time + '.mid')

    def load_model(self, load_path):

        # Loading SpatialPooler
        print 'Loading SpatialPooler'
        with open(load_path + 'sp.bin', 'rb') as sp:
            sp_builder = SpatialPoolerProto.read(
                sp, traversal_limit_in_words=2**61)
        self.network.regions['SpatialPooler'].getSelf(
        )._sfdr = self.network.regions['SpatialPooler'].getSelf()._sfdr.read(sp_builder)

        # Loading TemporalMemory
        print 'Loading TemporalMemory'
        self.network.regions['TemporalMemory'].getSelf().getAlgorithmInstance().loadFromFile(load_path+'tm.bin')

        # Loading end classifier
        print 'Loading duration classifier'
        with open(load_path + 'dcl.bin', 'rb') as dcl:
            dcl_builder = SdrClassifierProto.read(
                dcl, traversal_limit_in_words=2**61)
        self.duration_classifier = self.duration_classifier.read(dcl_builder)

        # Loading pitch classifier
        print 'Loading pitch classifier'
        with open(load_path + 'pcl.bin', 'rb') as pcl:
            pcl_builder = SdrClassifierProto.read(
                pcl, traversal_limit_in_words=2**61)
        self.pitch_classifier = self.pitch_classifier.read(pcl_builder)

        # Loading velocity classifier
        print 'Loading velocity classifier'
        with open(load_path + 'vcl.bin', 'rb') as vcl:
            vcl_builder = SdrClassifierProto.read(
                vcl, traversal_limit_in_words=2**61)
        self.velocity_classifier = self.velocity_classifier.read(vcl_builder)

    def save_model(self, save_path):

        # Saving SpatialPooler
        print 'Saving SpatialPooler'
        sp_builder = SpatialPoolerProto.new_message()
        self.network.regions['SpatialPooler'].getSelf(
        ).getAlgorithmInstance().write(sp_builder)
        with open(save_path + 'sp.bin', 'w+b') as sp:
            sp_builder.write(sp)

        # Saving TemporalMemory
        print 'Saving TemporalMemory'
        self.network.regions['TemporalMemory'].getSelf().getAlgorithmInstance().saveToFile(save_path+'tm.bin')

        # Saving end classifier
        print 'Saving duration classifier'
        dcl_builder = SdrClassifierProto.new_message()
        self.duration_classifier.write(dcl_builder)
        with open(save_path + 'dcl.bin', 'w+b') as dcl:
            dcl_builder.write(dcl)

        # Saving pitch classifier
        print 'Saving pitch classifier'
        pcl_builder = SdrClassifierProto.new_message()
        self.pitch_classifier.write(pcl_builder)
        with open(save_path + 'pcl.bin', 'w+b') as pcl:
            pcl_builder.write(pcl)

        # Saving velocity classifier
        print 'Saving velocity classifier'
        vcl_builder = SdrClassifierProto.new_message()
        self.velocity_classifier.write(vcl_builder)
        with open(save_path + 'vcl.bin', 'w+b') as vcl:
            vcl_builder.write(vcl)
