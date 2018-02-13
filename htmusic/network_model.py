
from nupic.engine import Network
from nupic.algorithms.sdr_classifier import SDRClassifier

import json
import numpy


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
        self.network.addRegion('EventEncoder', 'ScalarSensor', json.dumps(self.encoders_params['event']))
        self.network.addRegion('TickEncoder', 'ScalarSensor', json.dumps(self.encoders_params['tick']))
        self.network.addRegion('VelocityEncoder', 'ScalarSensor', json.dumps(self.encoders_params['velocity']))
        self.network.addRegion('PitchEncoder', 'ScalarSensor', json.dumps(self.encoders_params['pitch']))

        self.network.addRegion(
            'SpatialPooler', 'py.SPRegion', json.dumps(self.spatial_pooler_params))
        self.network.addRegion(
            'TemporalMemory', 'py.TMRegion', json.dumps(self.temporal_memory_params))

        # Creating inner classifiers for multifield prediction
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

    def train(self, event, tick, velocity, pitch, records_total):
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

    def load_model(self):
        pass

    def save_model(self, save_path):
        builder = SpatialPoolerProto.new_message()
        self.network.regions['SpatialPooler'].getSelf().write(builder)
        serializedMessage = builder.to_bytes_packed()
        print serializedMessage
