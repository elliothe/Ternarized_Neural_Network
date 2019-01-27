#### Models for CIFAR-10 ############
from .vanilla_resnet_cifar import resnet20, resnet32, resnet44, resnet56
from .tern_resnet_cifar import tern_resnet20, tern_resnet32, tern_resnet44, tern_resnet56

#### Models for ImageNet ############
from .alexnet_vanilla import alexnet_vanilla
from .alexnet_quan import tern_alexnet_ff_lf, tern_alexnet_fq_lq

from .ResNet_tern import resnet18b_ff_lf_tex1, resnet18b_fq_lq_tex1
from .ResNet_tern import resnet34b_ff_lf_tex1, resnet34b_fq_lq_tex1
from .ResNet_tern import resnet50b_ff_lf_tex1, resnet50b_fq_lq_tex1
from .ResNet_tern import resnet101b_ff_lf_tex1, resnet101b_fq_lq_tex1

from .ResNet_REL_tex2 import resnet18b_fq_lq_tern_tex_2
from .ResNet_REL_tex4 import resnet18b_fq_lq_tern_tex_4

