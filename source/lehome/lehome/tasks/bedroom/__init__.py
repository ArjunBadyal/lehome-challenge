import gymnasium as gym

from . import agents

gym.register(
    id="LeHome-BiSO101-Direct-Garment-v0",
    entry_point=f"{__name__}.garment_bi:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg:GarmentEnvCfg",
    },
)

gym.register(
    id="LeHome-SO101-Direct-Garment-v0",
    entry_point=f"{__name__}.garment:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_cfg:GarmentEnvCfg",
    },
)

gym.register(
    id="LeHome-BiSO101-Direct-Garment-v2",
    entry_point=f"{__name__}.garment_bi_v2:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg_v2:GarmentEnvCfg",
    },
)

gym.register(
    id="LeHome-BiSO101-Direct-Garment-SAC-v0",
    entry_point=f"{__name__}.garment_bi_v2:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg_v2:GarmentSACEnvCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

gym.register(
    id="LeHome-BiSO101-Direct-Garment-fling-v0",
    entry_point=f"{__name__}.garment_fling_bi:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_fling_bi_cfg:GarmentEnvCfg",
    },
)
