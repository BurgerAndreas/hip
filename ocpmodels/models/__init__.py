# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import BaseModel
from .cgcnn import CGCNN

# These models use torch_sparse
try:
    from .dimenet import DimeNetWrap as DimeNet
except ImportError as e:
    # print("Will not be able to use DimeNet. Error: ", e)
    pass

try:
    from .dimenet_plus_plus import DimeNetPlusPlusWrap as DimeNetPlusPlus
except ImportError as e:
    # print("Will not be able to use DimeNetPlusPlus. Error: ", e)
    pass

try:
    from .gemnet.gemnet import GemNetT
    from .gemnet_gp.gemnet import GraphParallelGemNetT as GraphParallelGemNetT
    from .gemnet_oc.gemnet_oc import GemNetOC
except ImportError as e:
    # print("Will not be able to use GemNet. Error: ", e)
    pass

from .forcenet import ForceNet
from .painn.painn import PaiNN
from .schnet import SchNetWrap as SchNet
from .scn.scn import SphericalChannelNetwork
from .spinconv import spinconv
