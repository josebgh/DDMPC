try:
    from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
except ImportError:
    print("Examples module does not exist. Importing configuration directly from current folder.")
    from configuration import *

try:
    pid_data = load_DataHandler("pid_data")
except FileNotFoundError:
    import os
    current_dir = os.getcwd()
    # Traverse up the directory tree by the specified number of levels
    for _ in range(3):
        current_dir = os.path.dirname(current_dir)
    pid_data = load_DataHandler("pid_data",folder=os.path.join(current_dir, r"stored_data/data"))


power_hp_TrainingData.add(pid_data)
power_hp_TrainingData.shuffle()
power_hp_TrainingData.split(0.8, 0.0, 0.2)


lin = LinearRegression()

power_hp_TrainingData.split(1.0, 0.0, 0.0)
lin.fit(training_data=power_hp_TrainingData)
power_hp_TrainingData.split(0.0, 0.0, 1.0)
lin.test(training_data=power_hp_TrainingData)

lin.print_coefficients(power_hp_TrainingData)

lin.save("power_hp_linReg", override=True)