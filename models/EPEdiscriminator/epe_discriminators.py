import logging
import math
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn import init
import torch.nn.functional as F
# import kornia as K


from math import sqrt



# this is for Kornia, used just for anti-aliased resizing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

logger = logging.getLogger(__name__)

class DomainNorm2d(nn.Module):
	def __init__(self, dim):
		super(DomainNorm2d, self).__init__()
		self._scale = nn.Parameter(torch.normal(1, 1, (1,dim,1,1)))
		self._bias  = nn.Parameter(torch.normal(0, 1, (1,dim,1,1)))

	def forward(self, x):
		return x.div(x.pow(2).sum(dim=1, keepdims=True).clamp(min=1e-5).sqrt()) * self._scale + self._bias

class CompareNorm2d(nn.Module):
	def __init__(self, dim):
		super(CompareNorm2d, self).__init__()
		self._scale  = nn.Parameter(torch.normal(1, 1, (1,dim,1,1)))
		self._bias   = nn.Parameter(torch.normal(0, 1, (1,dim,1,1)))
		self._reduce = nn.Sequential(nn.Conv2d(3*dim, dim, 1), nn.LeakyReLU(0.2, False))
		self._norm   = nn.InstanceNorm2d(dim, affine=False)

	def forward(self, x):
		z = self._norm(x)
		y = x.div(x.pow(2).sum(dim=1, keepdims=True).clamp(min=1e-5).sqrt())
		return self._reduce(torch.cat((x, y * self._scale + self._bias, z), 1))

class CompareNorm2d2(nn.Module):
	def __init__(self, dim):
		super(CompareNorm2d2, self).__init__()
		self._scale  = nn.Parameter(torch.normal(1, 1, (1,dim,1,1)))
		self._bias   = nn.Parameter(torch.normal(0, 1, (1,dim,1,1)))
		self._reduce = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Conv2d(3*dim, dim, 1), nn.LeakyReLU(0.2, False))
		self._norm   = nn.InstanceNorm2d(dim, affine=True)

	def forward(self, x):
		z = self._norm(x)
		y = x.div(x.pow(2).sum(dim=1, keepdims=True).clamp(min=1e-5).sqrt())
		return self._reduce(torch.cat((x, y * self._scale + self._bias, z), 1))


class DiscriminatorEnsemble(nn.Module):
	""" Wrap an ensemble of discriminators.
	"""

	def __init__(self, discs):
		"""
		discs -- iterable of networks
		"""

		super(DiscriminatorEnsemble, self).__init__()
		self._log  = logging.getLogger('epe.network.disc_ensemble') 
		self._log.debug(discs)
		self.discs = nn.ModuleList(discs)		
		pass

	def prepare_input(self, fix_input, run_discs, x):
		""" Prepare input for individual discriminators.

		This function needs take care of providing detached input for any of the discriminators if fix_input == True.
		It may save computation for all discriminators i with run_discs[i] == False as those will be ignored in forward and backward passes.

		fix_input -- detach input before providing it to individual discriminator.
		run_discs -- list of bool, indicates if discriminator should be run.
		x -- input from which to compute input for discriminators.
		"""
		raise NotImplementedError

	def forward(self, fix_input=False, run_discs=[], **x):
		""" Forward x through discriminators."""

		if type(run_discs) == bool:
			run_discs = [run_discs] * len(self.discs)
			pass

		assert len(run_discs) == len(self.discs)
		x = self.prepare_input(fix_input=fix_input, run_discs=run_discs, **x)
		return [di(xi) if rd else None for xi, rd, di in zip(x, run_discs, self.discs)]

	def __len__(self):
		return len(self.discs)


class ProjectionDiscriminator(nn.Module):
	def __init__(self, dim_in, dim_base, max_dim, num_layers=3, num_groups=8, num_strides=3,
				 dilate=False, no_out=False, cfg={}, hw=169):
		"""

		dim_in -- incoming channel width
		dim_base -- channel width after first convolution
		max_dim -- channel width is doubled every layer until max_dim
		num_layers -- number of convolutional layers
		norm -- batch, inst, group, spectral, domain, compare, compare2
		num_groups -- number of groups for group_norm
		num_strides -- how many layers should have stride 2 (counting from bottom)
		dilate -- increasing dilation per layer
		no_out -- no extra projection to channel width 1
		"""

		super(ProjectionDiscriminator, self).__init__()

		norm               = cfg.get('norm', 'group')
		
		self._log = logging.getLogger('epe.network.proj_disc') 
		self._log.debug(f'  Creating projection discriminator with {num_layers} layers and {norm} norm.')

		dims    = [dim_in] + [min(max_dim, dim_base*2**i) for i in range(num_layers)]
		strides = [2]*num_strides + [1] * (num_layers+1-num_strides)
		self.CGL = make_conv_layer(dims, strides, True, norm=='spectral', norm_factory[norm], False, 3)
		"""
		CGL
		"""
		dim_out = dims[3]

		if no_out:
			self.CLC = None
		else:
			self.CLC = nn.Sequential(nn.Conv2d(dim_out,dim_out,3,padding=1), nn.LeakyReLU(0.2, True), nn.Conv2d(dim_out, 1, 1))
		"""
		CLC
		"""
		self.num_layers = num_layers+1
		self.embedding = nn.Embedding(194,dim_out)
		##### 这里是embedding!!! 不是linear!!!!!
		self.num_layers = num_layers+1

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
			elif isinstance(m, nn.Embedding):
				nn.init.normal_(m.weight, std=0.01)
			elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.InstanceNorm2d)):
				try:
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
				except AttributeError:
					pass
				pass
			pass				
		pass


	def forward(self, t):
		p,l = t
		"""
		!!! t is a turple !!! p is the perceptual feature (xi) !! l is label map (robost_lab) !!
		"""
		batchsize = p.shape[0]
		self._log.debug(f'disc.forward(x: {p.shape}, y: {l.shape})')
		y = self.CGL(p)

		_,c,h,w = y.shape

		if l is not None:
			if l.dtype == torch.int64:
				# precomputed segmentation
				_, _, hy, wy = l.shape
				e = self.embedding(l.reshape(batchsize,-1))
				e = e.permute(0,2,1).reshape(batchsize,c,hy,wy)
				e = F.interpolate(e, (h, w), mode='bilinear', align_corners=True)
			else:
				e = F.interpolate(l, (h, w), mode='bilinear', align_corners=True)
				e = torch.argmax(torch.nn.functional.softmax(e, dim=1), axis=1, keepdims=True)
				e = e.reshape(batchsize,-1)
				e = self.embedding(e)
				e = e.permute(0,2,1).reshape(batchsize,c,h,w)
				pass

			if self.CLC is not None:
				e_y = (e * y).sum(dim=1,keepdims=True)
				z = self.CLC(y)
				s = z + e_y
			else:
				s = (e * y).sum(dim=1,keepdims=True)
		else:
			s = self.CLC(y)

		return s


def make_disc_backbones(configs, cfg):

	
	discs = []
	for i, c in enumerate(configs):
		(dim_in, dim_base, max_dim, num_layers, num_strides) = c
		discs.append(ProjectionDiscriminator(dim_in=dim_in, dim_base=dim_base, max_dim=max_dim, \
					num_layers=num_layers, num_strides=num_strides, dilate=False, \
					no_out=False, cfg=cfg, hw=(169 if i < 7 else 144)))
	return discs


# class PatchGANDiscriminator(DiscriminatorEnsemble):
# 	def __init__(self, cfg):
# 		self._parse_config(cfg)
#
# 		configs = [(3, 64, self._max_dim, self._num_layers, self._num_layers)] * self._num_discs
# 		super(PatchGANDiscriminator, self).__init__(make_disc_backbones(configs))##初始化父类DiscriminatorEnsemble
# 		self._log = logging.getLogger('epe.network.patchgan')
# 		self._log.debug(f'Discriminators: {self.discs}')
# 		pass
#
# 	def _parse_config(self, cfg):
# 		self._num_discs  = int(cfg.get('num_discs', 3))
# 		self._max_dim    = int(cfg.get('max_dim', 256))
# 		self._num_layers = int(cfg.get('num_layers', 5))
# 		self._norm       = cfg.get('norm', 'group')
# 		assert self._norm in ['group', 'spectral', 'inst', 'batch', 'domain', 'none', 'compare', 'compare2']
# 		pass
#
# 	def prepare_input(self, img, fix_input, run_discs, **kwargs):
# 		''' Creates an image pyramid from img.'''
# 		imgs = [(img, None)]
# 		for i in range(1, self.__len__()):
# 			imgi = torch.nn.functional.interpolate(\
# 				imgs[-1][0], scale_factor=0.5, mode='bilinear', align_corners=False)
# 			imgs.append((imgi.detach() if fix_input else imgi, None))
# 			pass
#
# 		return imgs


class PerceptualDiscEnsemble(DiscriminatorEnsemble):
	def __init__(self, cfg):

		self._parse_config(cfg)

		configs = [
			(64, min(self._max_dim, 64), self._max_dim, self._num_layers, 4), 
			(64, min(self._max_dim, 64), self._max_dim, self._num_layers, 4), 
			#
			(128, min(self._max_dim, 128), self._max_dim, self._num_layers, 3), 
			(128, min(self._max_dim, 128), self._max_dim, self._num_layers, 3), 
			#
			(256, min(self._max_dim, 256), self._max_dim, self._num_layers, 2), 
			(256, min(self._max_dim, 256), self._max_dim, self._num_layers, 2), 
			(256, min(self._max_dim, 256), self._max_dim, self._num_layers, 2), 
			#
			(512, min(self._max_dim, 512), self._max_dim, self._num_layers, 1), 
			(512, min(self._max_dim, 512), self._max_dim, self._num_layers, 1), 
			(512, min(self._max_dim, 512), self._max_dim, self._num_layers, 1)]

		super(PerceptualDiscEnsemble, self).__init__(make_disc_backbones(configs, cfg))
		self._log = logging.getLogger('epe.network.pde')
		self._log.debug(f'Discriminators: {self.discs}')
		pass

	def _parse_config(self, cfg={}):
		self._max_dim    = int(cfg.get('max_dim', 256))
		self._num_layers = int(cfg.get('num_layers', 5))
		self._norm       = cfg.get('norm', 'group')
		self._downsample = cfg.get('downsample', -1)

		assert self._norm in ['group', 'spectral', 'inst', 'batch', 'domain', 'none', 'compare', 'compare2']
		pass

	def prepare_input(self, *, vgg, img, fix_input, run_discs, **kwargs):
		""" Applies a VGG to img and returns feature maps from relu layers. """

		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'PDE:prepare(i:{img.shape}, fix:{fix_input}, run:{run_discs}, other: {kwargs})')
			pass
		#
		# if self._downsample > 0:
		# 	a = random.choice([1,2,4])
		# 	if a > 1:
		# 		#f = 1.0 / (1 + (self._downsample-1) * random.random())
		# 		img = K.geometry.rescale(img, 1.0/float(a), antialias=True)
		# 	pass
			
		xs = [(vgg.normalize(img), None)]

		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'  xs[0]:{xs[0][0].shape}')
			pass

		for i in range(self.__len__()):
			# assert xs[-1][0].shape[0] == 1
			xi = getattr(vgg, f'relu_{i}')(xs[-1][0])
			xs.append((xi.detach() if fix_input else xi, None))
			
			if self._log.isEnabledFor(logging.DEBUG):
				self._log.debug(f'  xs[{i+1}]:{xs[i+1][0].shape}')
				pass
			pass
		
		return xs[1:]


class PerceptualProjectionDiscEnsemble(PerceptualDiscEnsemble):
	def __init__(self, cfg={}):
		super(PerceptualProjectionDiscEnsemble, self).__init__(cfg)
		self._log = logging.getLogger('epe.network.ppde')
		pass

	def prepare_input(self, *, vgg, img, robust_labels, fix_input, run_discs, **kwargs):
		xs = super().prepare_input(vgg=vgg, img=img, fix_input=fix_input, run_discs=run_discs)
		return [(xi, robust_labels) for (xi,_) in xs]


#######################################
##############################################################################
#######################################
#######################################
#######################################
#######################################


norm_factory = {
	'none': None,
	'group': lambda d: nn.GroupNorm(8, d),
	'batch': lambda d: nn.BatchNorm2d(d, track_running_stats=False),
	'inst': lambda d: nn.InstanceNorm2d(d, affine=True, track_running_stats=False),
	'domain': lambda d: nn.DomainNorm(d),
}


def make_conv_layer(dims, strides=1, leaky_relu=True, spectral=False, norm_factory=None, skip_final_relu=False,
					kernel=3):
	""" Make simple convolutional networks without downsampling.

	dims -- list with channel widths, where len(dims)-1 is the number of concolutional layers to create.
	strides -- stride of first convolution if int, else stride of each convolution, respectively
	leaky_relu -- yes or no (=use ReLU instead)
	spectral -- use spectral norm
	norm_factory -- function taking a channel width and returning a normalization layer.
	skip_final_relu -- don't use a relu at the end
	kernel -- width of kernel
	"""

	if type(strides) == int:
		strides = [strides] + [1] * (len(dims) - 2)
		pass

	c = nn.Conv2d(dims[0], dims[1], kernel, stride=strides[0], bias=spectral)
	m = [] if kernel == 1 else [nn.ReplicationPad2d(kernel // 2)]
	m += [c if not spectral else torch.nn.utils.spectral_norm(c)]

	if norm_factory:
		m += [norm_factory(dims[1])]
		pass

	m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]

	num_convs = len(dims) - 2
	for i, di in enumerate(dims[2:]):
		c = nn.Conv2d(dims[i + 1], di, 3, stride=strides[i + 1], bias=spectral)

		if kernel > 1:
			m += [nn.ReplicationPad2d(kernel // 2)]
		m += [c if not spectral else torch.nn.utils.spectral_norm(c)]

		if norm_factory:
			m += [norm_factory(di)]
			pass

		if i == num_convs - 1 and skip_final_relu:
			continue
		else:
			m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]
		pass

	return nn.Sequential(*m)


class ResBlock(nn.Module):
	def __init__(self, dims, first_stride=1, leaky_relu=True, spectral=False, norm_factory=None, kernel=3):
		super(ResBlock, self).__init__()

		self.conv = make_conv_layer(dims, first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
		self.down = make_conv_layer([dims[0], dims[-1]], first_stride, leaky_relu, spectral, None, True, kernel=kernel) \
			if first_stride != 1 or dims[0] != dims[-1] else None
		self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
		pass

	def forward(self, x):
		return self.relu(self.conv(x) + (x if self.down is None else self.down(x)))


class Res2Block(nn.Module):
	def __init__(self, dims, first_stride=1, leaky_relu=True):
		super(Res2Block, self).__init__()

		self.conv = make_conv_layer(dims, first_stride, leaky_relu, False, None, False, kernel=3)
		self.down = make_conv_layer([dims[0], dims[-1]], first_stride, leaky_relu, False, None, True, kernel=1) \
			if first_stride != 1 or dims[0] != dims[-1] else None
		self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
		pass

	def forward(self, x):
		return 0.1 * self.conv(x) + (x if self.down is None else self.down(x))


class BottleneckBlock(nn.Module):
	def __init__(self, dim_in, dim_mid, dim_out, stride=1):
		super(BottleneckBlock, self).__init__()
		self._conv1 = nn.Conv2d(dim_in, dim_mid, 1)
		self._conv2 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(dim_mid, dim_mid, 3, stride=stride))
		self._conv3 = nn.Conv2d(dim_mid, dim_out, 1)
		self._relu = nn.LeakyReLU(0.2, True) if leaky_relu else nn.ReLU(True)
		self._norm1 = nn.GroupNorm(dim_mid)
		self._norm2 = nn.GroupNorm(dim_mid)
		self._norm3 = nn.GroupNorm(dim_out)
		self._down = nn.Conv2d(dim_in, dim_out, 1, stride=stride) if stride > 1 or dim_in != dim_out else None
		pass

	def forward(self, x):
		r = x if self_down is None else self._down(x)
		x = self._conv1(x)
		x = self._norm1(x)
		x = self._relu(x)
		x = self._conv2(x)
		x = self._norm2(x)
		x = self._relu(x)
		x = self._conv3(x)
		x = self._norm3(x)
		x = x + r
		x = self._relu(x)
		return x


class ResnextBlock(nn.Module):
	def __init__(self, dim_in, dim_mid, dim_out, groups=8, stride=1):
		super(ResnextBlock, self).__init__()
		self._conv1 = nn.Conv2d(dim_in, dim_mid, 1)
		self._conv2 = nn.Sequential(nn.ReplicationPad2d(1),
									nn.Conv2d(dim_mid, dim_mid, 3, stride=stride, groups=groups))
		self._conv3 = nn.Conv2d(dim_mid, dim_out, 1)
		self._relu = nn.LeakyReLU(0.2, True) if False else nn.ReLU(True)
		self._norm1 = nn.GroupNorm(groups, dim_mid)
		self._norm2 = nn.GroupNorm(groups, dim_mid)
		self._norm3 = nn.GroupNorm(groups, dim_out)
		self._down = nn.Conv2d(dim_in, dim_out, 1, stride=stride) if stride > 1 or dim_in != dim_out else None
		pass

	def forward(self, x):
		r = x if self._down is None else self._down(x)
		x = self._conv1(x)
		x = self._norm1(x)
		x = self._relu(x)
		x = self._conv2(x)
		x = self._norm2(x)
		x = self._relu(x)
		x = self._conv3(x)
		x = self._norm3(x)
		x = x + r
		x = self._relu(x)
		return x



