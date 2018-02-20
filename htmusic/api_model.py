
from nupic.algorithms.sdr_classifier import SDRClassifier
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from nupic.encoders.scalar import ScalarEncoder

import cPickle as pickle
import json
import numpy
import midi
from tqdm import tqdm

class HTMusicModel(object):

    def __init__(self, model_params):

        # Getting parameters for network regions
        self.spatial_pooler_params = model_params['SpatialPooler']
        self.temporal_memory_params = model_params['TemporalMemory']
        self.classifiers_params = model_params['Classifiers']
        self.encoders_params = model_params['Encoders']

        # Adding encoders
        self.event_encoder = ScalarEncoder(w=21, minval=128, maxval=144, n=500)
        self.tick_encoder = ScalarEncoder(w=21, minval=0, maxval=6000, n=500)
        self.velocity_encoder = ScalarEncoder(w=21, minval=0, maxval=128, n=500)
        self.pitch_encoder = ScalarEncoder(w=21, minval=0, maxval=128, n=500)

        # Getting encoding width
        self.encoding_width = self.event_encoder.getWidth() + \
                       self.tick_encoder.getWidth() + \
                       self.velocity_encoder.getWidth() + \
                       self.pitch_encoder.getWidth()
        
        # Adding Spatial Pooler
        self.spatial_pooler = SpatialPooler(inputDimensions = (self.spatial_pooler_params['inputWidth'],),
                                            columnDimensions = (self.spatial_pooler_params['columnCount'],),
                                            potentialRadius = self.spatial_pooler_params['potentialRadius'],
                                            potentialPct = self.spatial_pooler_params['potentialPct'],
                                            globalInhibition = self.spatial_pooler_params['globalInhibition'],
                                            localAreaDensity = self.spatial_pooler_params['localAreaDensity'],
                                            numActiveColumnsPerInhArea = self.spatial_pooler_params['numActiveColumnsPerInhArea'],
                                            stimulusThreshold = self.spatial_pooler_params['stimulusThreshold'],
                                            synPermInactiveDec = self.spatial_pooler_params['synPermInactiveDec'],
                                            synPermActiveInc = self.spatial_pooler_params['synPermActiveInc'],
                                            synPermConnected = self.spatial_pooler_params['synPermConnected'],
                                            minPctOverlapDutyCycle = self.spatial_pooler_params['minPctOverlapDutyCycle'],
                                            dutyCyclePeriod = self.spatial_pooler_params['dutyCyclePeriod'],
                                            boostStrength = self.spatial_pooler_params['boostStrength'],
                                            seed = self.spatial_pooler_params['seed'],
                                            spVerbosity = self.spatial_pooler_params['spVerbosity'],
                                            wrapAround = self.spatial_pooler_params['wrapAround'])
        
        
        # Adding Temporal Memory
        self.temporal_memory = TemporalMemory(columnDimensions = (self.temporal_memory_params['columnCount'],),
                                               cellsPerColumn = self.temporal_memory_params['cellsPerColumn'],
                                               activationThreshold = self.temporal_memory_params['activationThreshold'],
                                               initialPermanence = self.temporal_memory_params['initialPerm'],
                                               connectedPermanence = self.temporal_memory_params['connectedPermanence'],
                                               minThreshold = self.temporal_memory_params['minThreshold'],
                                               maxNewSynapseCount = self.temporal_memory_params['maxNewSynapseCount'],
                                               permanenceIncrement = self.temporal_memory_params['permanenceInc'],
                                               permanenceDecrement = self.temporal_memory_params['permanenceDec'],
                                               predictedSegmentDecrement = self.temporal_memory_params['predictedSegmentDecrement'],
                                               maxSegmentsPerCell = self.temporal_memory_params['maxSegmentsPerCell'],
                                               maxSynapsesPerSegment = self.temporal_memory_params['maxSynapsesPerSegment'],
                                               seed = self.temporal_memory_params['seed'])

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

    def train(self, event, tick, velocity, pitch, records_total):
        event_bits = numpy.zeros(self.event_encoder.getWidth())
        tick_bits = numpy.zeros(self.tick_encoder.getWidth())
        velocity_bits = numpy.zeros(self.velocity_encoder.getWidth())
        pitch_bits = numpy.zeros(self.pitch_encoder.getWidth())

        self.event_encoder.encodeIntoArray(event, event_bits)
        self.tick_encoder.encodeIntoArray(tick, tick_bits)
        self.velocity_encoder.encodeIntoArray(velocity, velocity_bits)
        self.pitch_encoder.encodeIntoArray(pitch, pitch_bits)

        encoding = numpy.concatenate(
            (event_bits, tick_bits, velocity_bits, pitch_bits)
        )
        
        active_columns = numpy.zeros(self.spatial_pooler_params['columnCount'])
        
        self.spatial_pooler.compute(encoding, True, active_columns)
        active_column_indicies = numpy.nonzero(active_columns)[0]
        self.temporal_memory.compute(active_column_indicies, learn=True)


        # Getting active cells of TM and bucket indicies of encoders to feed classifiers
        active_cells = self.temporal_memory.getActiveCells()

        event_bucket = self.event_encoder.getBucketIndices(event)
        tick_bucket = self.tick_encoder.getBucketIndices(tick)
        velocity_bucket = self.velocity_encoder.getBucketIndices(velocity)
        pitch_bucket = self.pitch_encoder.getBucketIndices(pitch)

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

    def generate(self, seed, records_total, output_dir):
        # Instantiate a MIDI Pattern (contains a list of tracks)
        pattern = midi.Pattern()
        # Instantiate a MIDI Track (contains a list of MIDI events)
        track = midi.Track()
        # Append the track to the pattern
        pattern.append(track)

        for iters in tqdm(range(records_total, records_total + 3000)):
            event = seed[0]
            tick = seed[1]
            velocity = seed[2]
            pitch = seed[3]

            event_bits = numpy.zeros(self.event_encoder.getWidth())
            tick_bits = numpy.zeros(self.tick_encoder.getWidth())
            velocity_bits = numpy.zeros(self.velocity_encoder.getWidth())
            pitch_bits = numpy.zeros(self.pitch_encoder.getWidth())

            self.event_encoder.encodeIntoArray(event, event_bits)
            self.tick_encoder.encodeIntoArray(tick, tick_bits)
            self.velocity_encoder.encodeIntoArray(velocity, velocity_bits)
            self.pitch_encoder.encodeIntoArray(pitch, pitch_bits)

            encoding = numpy.concatenate(
                [event_bits, tick_bits, velocity_bits, pitch_bits]
            )

            active_columns = numpy.zeros(self.spatial_pooler_params['columnCount'])

            self.spatial_pooler.compute(encoding, True, active_columns)
            active_column_indicies = numpy.nonzero(active_columns)[0]
            self.temporal_memory.compute(active_column_indicies, learn=True)


            # Getting active cells of TM and bucket indicies of encoders to feed classifiers
            active_cells = self.temporal_memory.getActiveCells()

            event_bucket = self.event_encoder.getBucketIndices(event)
            tick_bucket = self.tick_encoder.getBucketIndices(tick)
            velocity_bucket = self.velocity_encoder.getBucketIndices(velocity)
            pitch_bucket = self.pitch_encoder.getBucketIndices(pitch)

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


    def load_model(self, load_path):
        pass

    def save_model(self, save_path):
        with open(save_path + 'sp.tmp', 'wb') as sp:
            self.spatial_pooler.writeToFile(sp)
       
        with open(save_path + 'tm.tmp', 'wb') as tm:
            self.temporal_memory.writeToFile(tm)
    # def load_model(self, load_path):
    #     with open("sp.tmp", "w") as sp:
    #         self.spatial_pooler = SpatialPooler.readFromFile(sp)

    #     # with open("tm.pkl", "w") as tm:
    #     #     self.temporal_memory = pickle.load(tm)

    # def save_model(self, save_path):
    #     with open("sp.tmp", "wb") as sp:
    #         self.spatial_pooler.writeToFile(sp)

    #     # with open("tm.tmp", "wb") as tm:
    #     #     self.temporal_memory.writeToFile(tm)