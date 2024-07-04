try:
    from Examples.BopTest.bestest_hydronic_heat_pump.configuration import *
except ImportError:
    print("Examples module does not exist. Importing configuration directly from current folder.")
    from configuration import *

# regression for TAirRoom
TAirRoom_predictor: LinearRegression = load_LinearRegression(filename='TairRoom_linReg')
TAirRoom_predictor_SS = StateSpace(state_space=lr2ss(linear_regression = TAirRoom_predictor, model=model) , linear_regression = TAirRoom_predictor)

# TAirRoom_predictor: GaussianProcess = load_GaussianProcess(filename='TairRoom_GPR')
# TAirRoom_predictor: NeuralNetwork = load_NetworkTrainer(filename="TairRoom_ANN").best
TAirRoom.mode = Economic()


power_hp_predictor: LinearRegression = load_LinearRegression(filename='power_hp_linReg')
power_hp_predictor_SS = StateSpace(state_space=lr2ss(linear_regression = power_hp_predictor, model=model) , linear_regression = power_hp_predictor)

# power_hp_wb = WhiteBox(
#     inputs=[u_hp.source, t_amb.source, TAirRoom.source, u_hp_logistic.source],
#     output=power_hp,
#     output_expression=(
#         u_hp.source
#         * 10000
#         / ((TAirRoom.source + 15) / ((TAirRoom.source + 15) - t_amb.source) * 0.6)
#         + 1110 * u_hp_logistic.source
#     ),
#     step_size=one_minute * 15,
# )

# power_hp_gpr = load_GaussianProcess("power_hp_GPR_500_IP")

hhp_MPC = ModelPredictive(
    step_size=one_minute * 15,
    nlp=NLP(
        model=model,
        N=32,
        objectives=[
            Objective(feature=TAirRoom, cost=Quadratic(weight=50)),
            Objective(feature=costs_el, cost=AbsoluteLinear(weight=1)),
            # Objective(feature=u_hp,             cost=Linear(1)),
            Objective(feature=u_hp_change, cost=Quadratic(weight=0.1)),
        ],
        constraints=[
            Constraint(feature=u_hp, lb=0, ub=1),
        ],
    ),
    forecast_callback=system.get_forecast,
    solution_plotter=mpc_solution_plotter,
    show_solution_plot=False,
    save_solution_plot=False,
    save_solution_data=True,
    state_spaces = [TAirRoom_predictor_SS.state_space, power_hp_predictor_SS.state_space],
)

system.setup(
    start_time=one_week * 4,
    warmup_period=one_week,
    active_control_layers={"oveHeaPumY_activate": 1},
)

dh = load_DataHandler("pid_data")

TAirRoom_TrainingData.add(raw_data=dh)
power_hp_TrainingData.add(raw_data=dh)

solver_options = {
    "verbose": False,
    "ipopt.print_level": 1,
    "warn_initial_bounds": True,
    # 'ipopt.tol': 1e-5,
    # 'ipopt.acceptable_tol': 1e-2,
    # 'ipopt.acceptable_iter': 10,
}
system.run(controllers=(), duration=one_day)

# online mpc loop
for repetition in range(7):
    hhp_MPC.nlp.build(
        # solver_options=solver_options, predictors=[TAirRoom_predictor, power_hp_wb]
        solver_options=solver_options, predictors=[TAirRoom_predictor_SS, power_hp_predictor_SS]
    )


    hhp_MPC.setupMatlab(
        state_spaces = [TAirRoom_predictor_SS.state_space, power_hp_predictor_SS.state_space]
        # state_spaces = [TAirRoom_predictor_SS.state_space, Q_flowAhu_predictor_SS.state_space],
    )

    online_data = system.run(controllers=(hhp_MPC,), duration=one_day * 1)
    online_data.plot(plotter=mpc_plotter)

    # online learning TAirRoom
    TAirRoom_TrainingData.add(online_data)
    TAirRoom_TrainingData.shuffle()
    TAirRoom_TrainingData.split(0.7, 0.15, 0.15)
    TAirRoom_predictor.fit(
        training_data=TAirRoom_TrainingData,
        # epochs=100,
        # batch_size=50,
        # verbose=1,
    )
    TAirRoom_predictor.test(training_data=TAirRoom_TrainingData, show_plot=True)
    power_hp_TrainingData.add(online_data)
    power_hp_TrainingData.shuffle()
    power_hp_TrainingData.split(0.8, 0, 0.2)
    # power_hp_gpr.fit(training_data=power_hp_TrainingData)
    # power_hp_gpr.test(training_data=power_hp_TrainingData)
    power_hp_predictor.fit(training_data=power_hp_TrainingData)
    power_hp_predictor.test(training_data=power_hp_TrainingData)

    TAirRoom_predictor_SS = StateSpace(state_space=lr2ss(linear_regression = TAirRoom_predictor, model=model) , linear_regression = TAirRoom_predictor)
    power_hp_predictor_SS = StateSpace(state_space=lr2ss(linear_regression = power_hp_predictor, model=model) , linear_regression = power_hp_predictor)


    dh.save("mpc_linear_cost", override=True)
