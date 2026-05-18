from gymnasium.envs.registration import register
from .sme import SME

register(
    id="SME-v0",
    entry_point="sme_env.sme:SME",
)