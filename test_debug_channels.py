"""
Debug script to check channel dimensions and ZF computation
"""
import numpy as np
from envs.itsn_env import ITSNEnv
from envs.scenario import ITSNScenario
s = ITSNScenario()
c = s.generate_channels()
