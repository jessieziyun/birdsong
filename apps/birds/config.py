# CNN Version Laser Machine Listener
# Application configurations

from easydict import EasyDict

conf = EasyDict()

# Basic configurations
conf.sampling_rate = 16000
conf.duration = 1
conf.hop_length = 253 # to make time steps 64
conf.fmin = 20
conf.fmax = conf.sampling_rate // 2
conf.n_mels = 64
conf.n_fft = conf.n_mels * 20

# Labels
conf.labels = ['AfricanPiedWagtail', 'BarnSwallow', 'BlackWoodpecker', 'BlackheadedGull', 'CanadaGoose', 'CarrionCrow', 'CoalTit', 'CommonBlackbird', 'CommonChaffinch', 'CommonChiffchaff', 'CommonCuckoo', 'CommonHouseMartin', 'CommonLinnet', 'CommonMoorhen', 'CommonNightingale', 'CommonPheasant', 'CommonRedpoll', 'CommonRedshank', 'CommonRedstart', 'CommonReedBunting', 'CommonSnipe', 'CommonStarling', 'CommonSwift', 'CommonWhitethroat', 'CommonWoodPigeon', 'CornBunting', 'Dunlin', 'Dunnock', 'EurasianBlackcap', 'EurasianBlueTit', 'EurasianBullfinch', 'EurasianCollaredDove', 'EurasianCoot', 'EurasianGreenfinch', 'EurasianJay', 'EurasianMagpie', 'EurasianNuthatch', 'EurasianOystercatcher', 'EurasianReedWarbler', 'EurasianSkylark', 'EurasianTreeSparrow', 'EurasianTreecreeper', 'EurasianWren', 'EurasianWryneck', 'EuropeanBeeEater', 'EuropeanGoldenPlover', 'EuropeanGoldfinch', 'EuropeanGreenWoodpecker', 'EuropeanHerringGull', 'EuropeanHoneyBuzzard', 'EuropeanNightjar', 'EuropeanRobin', 'EuropeanTurtleDove', 'GardenWarbler', 'Goldcrest', 'GoldenOriole', 'GreatSpottedWoodpecker', 'GreatTit', 'GreyPartridge', 'GreyPlover', 'HouseSparrow', 'LesserWhitethroat', 'LongTailedTit', 'MarshTit', 'MarshWarbler', 'MeadowPipit', 'NorthernLapwing', 'NorthernRaven', 'RedCrossbil', 'RedthroatedLoon', 'Redwing', 'RiverWarbler', 'RockDove', 'Rook', 'SedgeWarbler', 'SongThrush', 'SpottedFlycatcher', 'StockDove', 'TawnyOwl', 'TreePipit', 'WesternJackdaw', 'WesternYellowWagtail', 'WillowPtarmigan', 'WillowTit', 'WillowWarbler', 'WoodSandpiper', 'WoodWarbler', 'Yellowhammer']

# Training configurations
conf.folder = '.'
conf.n_fold = 1
conf.normalize = 'samplewise'
conf.valid_limit = None
conf.random_state = 42
conf.samples_per_file = 30
conf.test_size = 0.2
conf.batch_size = 32
conf.learning_rate = 0.0001
conf.epochs = 50
conf.verbose = 2
conf.best_weight_file = 'best_model_weight.h5'
conf.eval_ensemble = False # This solution shuffles samples, ensemble not available

# Runtime conficurations
conf.rt_process_count = 1
conf.rt_oversamples = 10
conf.pred_ensembles = 10
conf.runtime_model_file = 'cnn-model.pb'
