from pip_library_etl import PipPlanner
import time
import pandas as pd

st = time.time()
functions = [
    pd.read_csv,
    pd.DataFrame.describe,
    pd.DataFrame.filter,
    pd.DataFrame.to_clipboard,
]

planner = PipPlanner(device="cloud")

planner.register_functions(functions)

plan = planner.generate_plan("read csv file at path 'a.csv' and describe it")


print(plan)
et = time.time()
print(et - st)
