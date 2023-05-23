from models.ganformer.NetworkS.networks import BipartiteDecoder
import torch

# mapping_kwargs = {'num_layers': 8,
#                   'layer_dim': None,
#                   'act': 'lrelu',
#                   'lrmul': 0.01,
#                   'w_avg_beta': 0.995,
#                   'resnet': True,
#                   'ltnt2ltnt': True,
#                   'transformer': True,
#                   'num_heads': 1,
#                   'attention_dropout': 0.12,
#                   'ltnt_gate': False,
#                   'use_pos': True,
#                   'normalize_global': True}
# synthesis_kwargs = {'crop_ratio': None,
#                     'channel_base': 32768,
#                     'channel_max': 512,
#                     'architecture': 'resnet',
#                     'resample_kernel': [1, 3, 3, 1],
#                     'local_noise': True,
#                     'act': 'lrelu',
#                     'ltnt_stem': False,
#                     'style': True,
#                     'transformer': True,
#                     'start_res': 0,
#                     'end_res': 8,
#                     'num_heads': 1,
#                     'attention_dropout': 0.12,
#                     'ltnt_gate': False,
#                     'img_gate': False,
#                     'integration': 'mul', 'norm': 'layer',
#                     'kmeans': True,
#                     'kmeans_iters': 1,
#                     'iterative': False,
#                     'use_pos': True,
#                     'pos_dim': None,
#                     'pos_type': 'sinus',
#                     'pos_init': 'uniform',
#                     'pos_directions_num': 2}
# _kwargs = {'nothing':None}


# G = bipartite_attention_computer(
#                 z_dim=32,                      # Input latent (Z) dimensionality
#                 c_dim=0,                      # Conditioning label (C) dimensionality
#                 w_dim=32,                      # Intermediate latent (W) dimensionality
#                 k=17,                          # Number of latent vector components z_1,...,z_k
#                 img_resolution=256,             # Output resolution
#                 img_channels=3,               # Number of output color channels
#                 component_dropout   = 0.0,  # Dropout over the latent components, 0 = disable
#                 mapping_kwargs      = mapping_kwargs,
#                 synthesis_kwargs    = synthesis_kwargs,   # Arguments for SynthesisNetwork
#                 **_kwargs                   # Ignore unrecognized keyword args
#              )
#
# batch_size = 2
# z = torch.randn([batch_size, *G.input_shape[1:]], device = 'cpu')         # Sample latent vector
# input_for_decoder = G(z, truncation_psi = 0.7 )
# print(input_for_decoder)







# G = BipartiteEncoder(
#                 z_dim=32,                      # Input latent (Z) dimensionality
#                 c_dim=0,                      # Conditioning label (C) dimensionality
#                 w_dim=32,                      # Intermediate latent (W) dimensionality
#                 k=17,                          # Number of latent vector components z_1,...,z_k
#                 img_resolution=256,             # Output resolution
#                 img_channels=3,               # Number of output color channels
#                 component_dropout   = 0.0,  # Dropout over the latent components, 0 = disable
#                 mapping_kwargs      = mapping_kwargs,
#                 synthesis_kwargs    = synthesis_kwargs,   # Arguments for SynthesisNetwork
#                 **_kwargs                   # Ignore unrecognized keyword args
#              )
#
#
# batch_size = 4
# # label = torch.randn(batch_size,67,256,512)
# label = torch.randn(batch_size,67,256,512)
# z = torch.randn([batch_size, *G.input_shape[1:]], device = 'cpu')         # Sample latent vector
# input_for_decoder = G(label,z, truncation_psi = 0.7 )
# print(input_for_decoder)




mapping_kwargs = {'num_layers': 8,
                  'layer_dim': None,
                  'act': 'lrelu',
                  'lrmul': 0.01,
                  'w_avg_beta': 0.995,
                  'resnet': True,
                  'ltnt2ltnt': True,
                  'transformer': True,
                  'num_heads': 1,
                  'attention_dropout': 0.12,
                  'ltnt_gate': False,
                  'use_pos': True,
                  'normalize_global': True}
decoder_kwargs = {'crop_ratio': None,
                    'channel_base': 32768,
                    'channel_max': 512,
                    'architecture': 'resnet',
                    'resample_kernel': [1, 3, 3, 1],
                    'local_noise': True,
                    'act': 'lrelu',
                    'ltnt_stem': False,
                    'style': True,
                    'transformer': True,
                    'start_res': 0,
                    'end_res': 8,
                    'num_heads': 1,
                    'attention_dropout': 0.12,
                    'ltnt_gate': False,
                    'img_gate': False,
                    'integration': 'mul',
                    'norm': 'layer',
                    'kmeans': True,
                    'kmeans_iters': 1,
                    'iterative': False,
                    'use_pos': True,
                    'pos_dim': None,
                    'pos_type': 'sinus',
                    'pos_init': 'uniform',
                    'pos_directions_num': 2}
_kwargs = {'nothing':None}




G = BipartiteDecoder(
                z_dim=32,                      # Input latent (Z) dimensionality
                c_dim=0,                      # Conditioning label (C) dimensionality
                w_dim=32,                      # Intermediate latent (W) dimensionality
                k=17,                          # Number of latent vector components z_1,...,z_k
                img_resolution=256,             # Output resolution
                img_channels=3,               # Number of output color channels
                component_dropout   = 0.0,  # Dropout over the latent components, 0 = disable
                mapping_kwargs      = mapping_kwargs,
                decoder_kwargs    = decoder_kwargs,   # Arguments for SynthesisNetwork
                **_kwargs                   # Ignore unrecognized keyword args
             )


batch_size = 3
# label = torch.randn(batch_size,67,256,512)
label = None

from_encoder = {
    'res16':[torch.randn(batch_size,512,16,32),torch.randn(batch_size,512,16,32),torch.randn(batch_size,512,16,32),],
    'res32':[torch.randn(batch_size,256,32,64),torch.randn(batch_size,256,32,64),torch.randn(batch_size,256,32,64),],
    'res64':[torch.randn(batch_size,128,64,128),torch.randn(batch_size,128,64,128),torch.randn(batch_size,128,64,128),],
    'res128':[torch.randn(batch_size,64,128,256),torch.randn(batch_size,64,128,256),torch.randn(batch_size,64,128,256),],
    'res256':[torch.randn(batch_size,32,256,512),torch.randn(batch_size,32,256,512),torch.randn(batch_size,32,256,512),
              torch.randn(batch_size,32,256,512),torch.randn(batch_size,32,256,512),]
}
emb = {
    'res16':[torch.randn(batch_size,512,16,32),torch.randn(batch_size,512,16,32)],
    'res32':[torch.randn(batch_size,256,32,64),torch.randn(batch_size,256,32,64)],
    'res64':[torch.randn(batch_size,128,64,128),torch.randn(batch_size,128,64,128)],
    'res128':[torch.randn(batch_size,64,128,256),torch.randn(batch_size,64,128,256)],
    'res256':[torch.randn(batch_size,32,256,512),torch.randn(batch_size,32,256,512)]
}

z = torch.randn([batch_size, *G.input_shape[1:]], device = 'cpu')         # Sample latent vector
output_from_decoder,img = G(from_encoder,emb,label,z, truncation_psi = 0.7 )
print(output_from_decoder)


