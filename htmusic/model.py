
from nupic.engine import Network
from nupic.algorithms.sdr_classifier import SDRClassifier

import json


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
		self.network.addRegion('EventEncoder','ScalarSensor', json.dumps(self.encoders_params['event']))
		self.network.addRegion('TickEncoder','ScalarSensor', json.dumps(self.encoders_params['tick']))
		self.network.addRegion('VelocityEncoder','ScalarSensor', json.dumps(self.encoders_params['velocity']))
		self.network.addRegion('PitchEncoder','ScalarSensor', json.dumps(self.encoders_params['pitch']))

		self.network.addRegion('SpatialPooler', 'py.SPRegion', json.dumps(self.spatial_pooler_params))
		self.network.addRegion('TemporalMemory', 'py.TMRegion', json.dumps(self.temporal_memory_params))

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
		self.network.link('VelocityEncoder', 'SpatialPooler', 'UniformLink', '')
		self.network.link('PitchEncoder', 'SpatialPooler', 'UniformLink', '')
		self.network.link('SpatialPooler', 'TemporalMemory', 'UniformLink', '', srcOutput='bottomUpOut', destInput='bottomUpIn')

	def _enable_learning(self):
		# Enable learning for all regions.
		self.network.regions["SpatialPooler"].setParameter("learningMode", 1)
		self.network.regions["TemporalMemory"].setParameter("learningMode", 1)

	def _enable_inference(self):
		# Enable inference for all regions.
		self.network.regions["SpatialPooler"].setParameter("inferenceMode", 1)
		self.network.regions["TemporalMemory"].setParameter("inferenceMode", 1)

	def load_model(self):
		pass

	def save_model(self, save_path):
		builder = SpatialPoolerProto.new_message()
		self.network.regions['SpatialPooler'].getSelf().write(builder)
		serializedMessage = builder.to_bytes_packed()
		print serializedMessage

