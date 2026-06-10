from __future__ import annotations

from typing import Dict, Tuple, Type

from .client import FedALAClient, FedASClient, FedAvgClient, FedL2PClient, FedProxClient, TARGETClient, TagFedClient, FedDERPPClient, FedEWCClient, FedLwFClient
from .server import FedALAServer, FedASServer, FedAvgServer, FedL2PServer, FedProxServer, TARGETServer, TagFedServer

from .client import AFFCLClient
from .server import AFFCLServer

from .client import DDDRClient
from .server import DDDRServer



METHOD_REGISTRY: Dict[str, Tuple[Type, Type]] = {
    "fedavg": (FedAvgServer, FedAvgClient),
    "fedprox": (FedProxServer, FedProxClient),
    "fedala": (FedALAServer, FedALAClient),
    "fedas": (FedASServer, FedASClient),
    "fedl2p": (FedL2PServer, FedL2PClient),
    "target": (TARGETServer, TARGETClient),
    "tagfed": (TagFedServer, TagFedClient),
    "fedewc": (FedAvgServer, FedEWCClient),
    "fedlwf": (FedAvgServer, FedLwFClient),
    "fedderpp": (FedAvgServer, FedDERPPClient),
    "affcl": (AFFCLServer, AFFCLClient),
    "dddr": (DDDRServer, DDDRClient),
}


def build_method(name: str):
    if name not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Available methods: {sorted(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name]


def list_methods() -> list[str]:
    return sorted(METHOD_REGISTRY)