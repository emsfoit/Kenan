import simpy
import pandas as pd
import pybamm

import sys


def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    print(f'size={size}')


def clean_data(file_path):  # cleaning the workload data

    with open(file_path, "r") as csvfile:
        df_t = pd.read_csv(csvfile, index_col=0, parse_dates=True, dtype='unicode')

    df_t = df_t.reset_index(drop=True)
    df_t = df_t.drop(
        ['passenger_count', 'RatecodeID', 'store_and_fwd_flag', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
         'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge'], axis=1)
    df_t['tpep_pickup_datetime'] = pd.to_datetime(df_t['tpep_pickup_datetime'])
    df_t = df_t[~(df_t['tpep_pickup_datetime'] < '2020-06-01 00:00:00')]
    df_t = df_t[~(df_t['tpep_pickup_datetime'] > '2020-07-01 00:00:00')]
    df_t["c"] = 1
    # group the pickup action every 15 minutes
    df_t = df_t.groupby(pd.Grouper(freq='15min', key='tpep_pickup_datetime')).sum()["c"]
    df_t = df_t.to_frame()

    return df_t


# last_solution=None
discharge_capacity_rwa = []


class Battery:  # check the power drwan/feeded to the grid

    def __init__(self, env, capacity, charge_level=0):
        self.env = env

        self.last_update = env.now
        self.last_solution = None

        self.parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        self.model = pybamm.lithium_ion.DFN()

        # TODO use these to initi
        self.charge_level = charge_level
        self.capacity = capacity
        self.soc = 70

    def update(self, power):
        global last_solution
        global soc
        grid_energy = 0
        c = 0  # extracharge
        i = 0
        capcity_factor = self.capacity / 5
        if self.soc < 94 and power > 0:
            charge_new = power / capcity_factor
            if charge_new < (95 - self.soc):
                experiment = pybamm.Experiment([f"charge at {power / capcity_factor} W for 15 m"])
            else:
                grid_energy = (charge_new - (95 - self.soc)) * 5
                print(f'grid_energy={grid_energy}')
                print(f'c={c}')
                print('charge power/capcity_factor')

                experiment = pybamm.Experiment([f"charge at {(95 - self.soc)} W for 15 m"])
                # experiment = pybamm.Experiment([f"charge at {(95-self.soc)*0.5} W for 15 m"])

                print('charge (95-self.soc)*1.22')


        elif self.soc > 15 and power < 0:
            experiment = pybamm.Experiment([f"discharge at {- power / capcity_factor} W for 15 m"])
            print('discharge')
        elif self.soc < 15 and power < 0:
            experiment = pybamm.Experiment(["Rest for 15 m"])
            print('Rest')
            grid_energy = power
        else:
            experiment = pybamm.Experiment(["Rest for 15 m"])
            print('Rest')
            grid_energy = power
        # time_passed = self.env.now - self.last_update

        i = +1

        sim = pybamm.Simulation(self.model, parameter_values=self.parameter_values, experiment=experiment)
        self.last_solution = sim.solve(starting_solution=self.last_solution, save_at_cycles=i)
        print(f'len(self.last_solution.cycles):{len(self.last_solution.cycles)}')
        # sim.solve(starting_solution=last_solution)
        # last_solution = sim.solution

        # self.last_solution = sim.solution
        # print(sim.solution)
        d = self.last_solution['Discharge capacity [A.h]']
        self.charge_level = self.charge_level + d.entries[-1]
        self.soc = ((5 - d.entries[-1]) / 5) * 100
        # print(f'State of charge={self.soc}%')
        sizeof(d.entries)
        #sim.plot(['Discharge capacity [A.h]'])
        # print(f'len(solution.all_ys)={len(self.last_solution.all_ys)}')
        # self.last_solution.all_ys=self.last_solution.all_ys[-15:]
        # self.last_solution.all_ts=self.last_solution.all_ts[-15:]
        # self.last_solution.all_models=self.last_solution.all_models[-15:]
        # self.last_solution.all_inputs=self.last_solution.all_inputs[-15:]
        del self.last_solution.all_ys[:][:-1]
        del self.last_solution.all_ts[:][:-1]
        del self.last_solution.all_inputs[:][:-1]
        del self.last_solution.all_models[:][:-1]
        # del self.last_solution.y[:-1][:-1]
        # solution.all_ys[-1][-1]
        discharge_capacity_rwa.append(d.entries[-1])

        return grid_energy


DELTA_ENERGY = []
GRID_ENEERGY = []
CHARGE_LEVEL = []


def simulate(env, battery, production_df, consumption_df):
    for i in range(0, len(production_df)):
        # print(f'i={i}')
        delta_energy = production_df.iloc[i ,0] - consumption_df.iloc[i , 0]
        grid_energy = battery.update(delta_energy)
        # print(f'grid={grid_energy}')
        # print(f'i={i}', f'delta_energy={delta_energy}')
        # print('-'*30)
        DELTA_ENERGY.append(delta_energy)
        GRID_ENEERGY.append(grid_energy)
        CHARGE_LEVEL.append(battery.charge_level)
        yield env.timeout(1)
    

def dataframe(df_w, delta_energy, grid_energy, charge_level,discharge_capacity_rwa):
    data = {'DELTA_ENERGY': delta_energy, 'GRID_ENERGY': grid_energy, 'CHARGE_LEVEL': charge_level,'DISCHARGE_CAPACITY':discharge_capacity_rwa}
    dataframe = pd.DataFrame(data, index=df_w.index)
    return dataframe


def main(solar_area, load_factor, capacity):
    # reading the Weather data and drop the index and the unneeded columns

    solar_efficiency = 0.18

    with open("./dataset/ms.txt","r") as csvfile:
        df_w = pd.read_csv(csvfile, index_col=0, parse_dates=True)  # W/m^2
    production_df = df_w * solar_area * solar_efficiency  # W

    # reading the Taxi data and drop the index and the unneeded columns
    df_t = clean_data("./dataset/yellow1.csv")
    # print(df_t)

    consumption_df = df_t * load_factor

    env = simpy.Environment()
    battery = Battery(env, capacity=capacity)
    # battery = Battery(env)
    env.process(simulate(env, battery, production_df, consumption_df))
    env.run()

    result = dataframe(df_w, DELTA_ENERGY, GRID_ENEERGY, CHARGE_LEVEL,discharge_capacity_rwa)
    with open("result.csv", "w") as f:
        f.write(result.to_csv())
    print(result)
    return result
    

if __name__ == '__main__':
    main(5, 1.05, 200)
  

#figure(figsize=(10.7),dip=80)
#pyplot.plot(discharge_capacity_rwa)