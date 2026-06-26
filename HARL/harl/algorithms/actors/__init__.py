"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.mae_happo import MAE_HAPPO
from harl.algorithms.actors.mae_hatrpo import MAE_HATRPO
from harl.algorithms.actors.mae_mappo import MAE_MAPPO

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "mappo": MAPPO,
    "MAE_happo": MAE_HAPPO,
    "MAE_hatrpo": MAE_HATRPO,
    "MAE_mappo": MAE_MAPPO,
}
