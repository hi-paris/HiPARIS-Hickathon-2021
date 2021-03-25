import sys
import pickle
import gym
import numpy as np
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv


class NormalizedMicroGridEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.parameters = env.mg.get_parameters()
        for key in self.parameters.columns:
            print(key, self.parameters[key][0])
        self.max_load = self.parameters.load[0]
        self.max_pv = self.parameters["PV_rated_power"][0]
        self.max_capa = self.parameters["battery_capacity"][0]
        self.max_price_import = self.parameters["grid_power_import"][0]
        self.max_price_export = self.parameters["grid_power_export"][0]
        self.max_hour = 23
        self.norm = 1. / np.array([self.max_load, self.max_hour, self.max_pv, 1., self.max_capa, self.max_capa, 1., 1., self.max_price_import, self.max_price_export], dtype=np.float32)
        self._cumul_hour = 0
        self._hours_per_week = 24*7
        self._hours_per_year = 8737

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)

    def reset(self):
        self._cumul_hour = 0
        return super().reset()

    def observation(self, observation):
        """
        original observation has shape (10,)
            [load, hour, pv, battery_soc, capa_to_charge, capa_to_discharge,
                            grid_status, grid_co2, grid_price_import, grid_price_export]

        output observation has
            [day progress (0..1,),  -> tells us the progress of the day
            week progress (0 .. 1)  -> tells us the progress of the week (like 0.8 is the start of the weekend)
            year progress (0..1)    -> progress of the year
            normalized load ()
            pv norm (), normalized capacity, grid price import (), grid price export ? (can the other buildings export ?)]
        """
        #assert len(observation) == 10
        progress_week = (self._cumul_hour % self._hours_per_week) / self._hours_per_week
        progress_year = self._cumul_hour / self._hours_per_year
        self._cumul_hour += 1
        return np.concatenate([np.float32(observation) * self.norm, np.array([progress_week, progress_year], dtype=np.float32)])


if __name__ == "__main__":
    with open('building_1.pkl', 'rb') as f:
        building_1 = pickle.load(f)
    with open('building_2.pkl', 'rb') as f:
        building_2 = pickle.load(f)
    with open('building_3.pkl', 'rb') as f:
        building_3 = pickle.load(f)

    buildings = [building_1, building_2, building_3]

    env = MicroGridEnv(
        env_config={'microgrid': buildings[2], "testing": False})

    env = NormalizedMicroGridEnv(env)
