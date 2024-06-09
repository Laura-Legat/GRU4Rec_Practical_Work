# baseline configuration
from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'bpr-max'),
('constrained_embedding', True),
('embedding', 0),
('elu_param', 1),
('layers', [64]),
('n_epochs', 10),
('batch_size', 50),
('dropout_p_embed', 0.4),
('dropout_p_hidden', 0.2),
('learning_rate', 0.05),
('momentum', 0.3),
('n_sample', 2048),
('sample_alpha', 0.3),
('bpreg', 0.9),
('logq', 0.0)
])
