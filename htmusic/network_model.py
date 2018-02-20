import capnp
import cPickle as pickle
from nupic.engine import Network
from nupic.algorithms.backtracking_tm_cpp_capnp import BacktrackingTMCppProto
from nupic.proto.SdrClassifier_capnp import SdrClassifierProto
from nupic.proto.SpatialPoolerProto_capnp import SpatialPoolerProto
from nupic.algorithms.sdr_classifier import SDRClassifier
from tqdm import tqdm

import random
import json
import numpy
import midi
import os


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
        self.network.addRegion('EventEncoder', 'ScalarSensor', json.dumps(
            self.encoders_params['event']))
        self.network.addRegion('TickEncoder', 'ScalarSensor',
                               json.dumps(self.encoders_params['tick']))
        self.network.addRegion('VelocityEncoder', 'ScalarSensor', json.dumps(
            self.encoders_params['velocity']))
        self.network.addRegion('PitchEncoder', 'ScalarSensor', json.dumps(
            self.encoders_params['pitch']))

        self.network.addRegion(
            'SpatialPooler', 'py.SPRegion', json.dumps(self.spatial_pooler_params))
        self.network.addRegion(
            'TemporalMemory', 'py.TMRegion', json.dumps(self.temporal_memory_params))

        # Creating outer classifiers for multifield prediction
        eclp = self.classifiers_params['event']
        tclp = self.classifiers_params['tick']
        vclp = self.classifiers_params['velocity']
        pclp = self.classifiers_params['pitch']

        self.event_classifier = SDRClassifier(steps=(1,), verbosity=eclp['verbosity'], alpha=eclp['alpha'],
                                              actValueAlpha=eclp['actValueAlpha'])
        self.tick_classifier = SDRClassifier(steps=(1,), verbosity=tclp['verbosity'], alpha=tclp['alpha'],
                                             actValueAlpha=tclp['actValueAlpha'])
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
        self.network.link('EventEncoder', 'SpatialPooler', 'UniformLink', '')
        self.network.link('TickEncoder', 'SpatialPooler', 'UniformLink', '')
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

    def train(self, event, tick, velocity, pitch):
        records_total = self.network.regions['SpatialPooler'].getSelf().getAlgorithmInstance().getIterationNum()
        self.network.regions['EventEncoder'].setParameter(
            'sensedValue', event)
        self.network.regions['TickEncoder'].setParameter(
            'sensedValue', tick)
        self.network.regions['VelocityEncoder'].setParameter(
            'sensedValue', velocity)
        self.network.regions['PitchEncoder'].setParameter(
            'sensedValue', pitch)
        self.network.run(1)

        # Getting active cells of TM and bucket indicies of encoders to feed classifiers
        active_cells = numpy.array(
            self.network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]
        event_bucket = numpy.array(
            self.network.regions['EventEncoder'].getOutputData('bucket'))
        tick_bucket = numpy.array(
            self.network.regions['TickEncoder'].getOutputData('bucket'))
        velocity_bucket = numpy.array(
            self.network.regions['VelocityEncoder'].getOutputData('bucket'))
        pitch_bucket = numpy.array(
            self.network.regions['PitchEncoder'].getOutputData('bucket'))

        # Getting up classifiers result
        event_classifier_result = self.event_classifier.compute(
            recordNum=records_total,
            patternNZ=active_cells,
            classification={
                'bucketIdx': event_bucket[0],
                'actValue': event
            },
            learn=True,
            infer=False
        )

        tick_classifier_result = self.tick_classifier.compute(
            recordNum=records_total,
            patternNZ=active_cells,
            classification={
                'bucketIdx': tick_bucket[0],
                'actValue': tick
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

    def generate(self, seed, output_dir, event_amount):
        records_total = self.network.regions['SpatialPooler'].getSelf().getAlgorithmInstance().getIterationNum()
        # Instantiate a MIDI Pattern (contains a list of tracks)
        pattern = midi.Pattern()
        # Instantiate a MIDI Track (contains a list of MIDI events)
        track = midi.Track()
        # Append the track to the pattern
        pattern.append(track)

        for iters in tqdm(range(records_total, records_total + event_amount)):
            event = seed[0]
            tick = seed[1]
            velocity = seed[2]
            pitch = seed[3]

            self.network.regions['EventEncoder'].setParameter(
                'sensedValue', event)
            self.network.regions['TickEncoder'].setParameter(
                'sensedValue', tick)
            self.network.regions['VelocityEncoder'].setParameter(
                'sensedValue', velocity)
            self.network.regions['PitchEncoder'].setParameter(
                'sensedValue', pitch)
            self.network.run(1)

            # Getting active cells of TM and bucket indicies of encoders to feed classifiers
            active_cells = numpy.array(
                self.network.regions['TemporalMemory'].getOutputData('bottomUpOut')).nonzero()[0]
            event_bucket = numpy.array(
                self.network.regions['EventEncoder'].getOutputData('bucket'))
            tick_bucket = numpy.array(
                self.network.regions['TickEncoder'].getOutputData('bucket'))
            velocity_bucket = numpy.array(
                self.network.regions['VelocityEncoder'].getOutputData('bucket'))
            pitch_bucket = numpy.array(
                self.network.regions['PitchEncoder'].getOutputData('bucket'))

            # Getting up classifiers result
            event_classifier_result = self.event_classifier.compute(
                recordNum=records_total,
                patternNZ=active_cells,
                classification={
                    'bucketIdx': event_bucket[0],
                    'actValue': event
                },
                learn=False,
                infer=True
            )

            tick_classifier_result = self.tick_classifier.compute(
                recordNum=records_total,
                patternNZ=active_cells,
                classification={
                    'bucketIdx': tick_bucket[0],
                    'actValue': tick
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

            ev = event_classifier_result[1].argmax()
            ti = tick_classifier_result[1].argmax()
            ve = velocity_classifier_result[1].argmax()
            pi = pitch_classifier_result[1].argmax()

            if event_classifier_result['actualValues'][ev] == 144:
                midi_event = midi.NoteOnEvent(tick=int(tick_classifier_result['actualValues'][ti]),
                                              velocity=int(velocity_classifier_result['actualValues'][ve]),
                                              pitch=int(pitch_classifier_result['actualValues'][pi]))

            else:
                midi_event = midi.NoteOffEvent(tick=int(tick_classifier_result['actualValues'][ti]),
                                               velocity=int(0),
                                               pitch=int(pitch_classifier_result['actualValues'][pi]))

            track.append(midi_event)

            seed = [event_classifier_result['actualValues'][ev], tick_classifier_result['actualValues'][ti],
                    velocity_classifier_result['actualValues'][ve], pitch_classifier_result['actualValues'][pi]]
            
        # Add the end of track event, append it to the track
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
        # Save the pattern to disk
        midi.write_midifile(output_dir + 'example.mid', pattern)

    def load_model(self, load_path):
        
        # Loading SpatialPooler
        print 'Loading SpatialPooler'
        with open(load_path + 'sp.bin', 'rb') as sp:
            sp_builder = SpatialPoolerProto.read(sp, traversal_limit_in_words=2**61)
        self.network.regions['SpatialPooler'].getSelf()._sfdr = self.network.regions['SpatialPooler'].getSelf()._sfdr.read(sp_builder)

        # Loading TemporalMemory
        print 'Loading TemporalMemory'
        self.network.regions['TemporalMemory'].getSelf().getAlgorithmInstance().loadFromFile(load_path+'tm.bin')

        # Loading event classifier
        print 'Loading event classifier'
        with open(load_path + 'ecl.bin', 'rb') as ecl:
            ecl_builder = SdrClassifierProto.read(ecl, traversal_limit_in_words=2**61)
        self.event_classifier = self.event_classifier.read(ecl_builder)
        
        # Loading tick classifier
        print 'Loading tick classifier'
        with open(load_path + 'tcl.bin', 'rb') as tcl:
            tcl_builder = SdrClassifierProto.read(tcl, traversal_limit_in_words=2**61)
        self.tick_classifier = self.tick_classifier.read(tcl_builder)

        # Loading velocity classifier
        print 'Loading velocity classifier'
        with open(load_path + 'vcl.bin', 'rb') as vcl:
            vcl_builder = SdrClassifierProto.read(vcl, traversal_limit_in_words=2**61)
        self.velocity_classifier = self.velocity_classifier.read(vcl_builder)

        # Loading pitch classifier
        print 'Loading pitch classifier'
        with open(load_path + 'pcl.bin', 'rb') as pcl:
            pcl_builder = SdrClassifierProto.read(pcl, traversal_limit_in_words=2**61)
        self.pitch_classifier = self.pitch_classifier.read(pcl_builder)


    def save_model(self, save_path):

        # Saving SpatialPooler
        print 'Saving SpatialPooler'
        sp_builder = SpatialPoolerProto.new_message()
        self.network.regions['SpatialPooler'].getSelf().getAlgorithmInstance().write(sp_builder)
        with open(save_path + 'sp.bin', 'w+b') as sp:
            sp_builder.write(sp)

        # Saving TemporalMemory
        print 'Saving TemporalMemory'
        self.network.regions['TemporalMemory'].getSelf().getAlgorithmInstance().saveToFile(save_path+'tm.bin')

        # Saving event classifier
        print 'Saving event classifier'
        ecl_builder = SdrClassifierProto.new_message()
        self.event_classifier.write(ecl_builder)
        with open(save_path + 'ecl.bin', 'w+b') as ecl:
            ecl_builder.write(ecl)

        # Saving tick classifier
        print 'Saving tick classifier'
        tcl_builder = SdrClassifierProto.new_message()
        self.tick_classifier.write(tcl_builder)
        with open(save_path + 'tcl.bin', 'w+b') as tcl:
            tcl_builder.write(tcl)

        # Saving velocity classifier
        print 'Saving velocity classifier'
        vcl_builder = SdrClassifierProto.new_message()
        self.velocity_classifier.write(vcl_builder)
        with open(save_path + 'vcl.bin', 'w+b') as vcl:
            vcl_builder.write(vcl)

        # Saving pitch classifier
        print 'Saving pitch classifier'
        pcl_builder = SdrClassifierProto.new_message()
        self.pitch_classifier.write(pcl_builder)
        with open(save_path + 'pcl.bin', 'w+b') as pcl:
            pcl_builder.write(pcl)