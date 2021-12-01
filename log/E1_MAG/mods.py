from multiagent_gridworld import *

def all_observation_moving_actions():
    for observation in all_observations():
        for moving_action in all_moving_actions():
            yield (observation, moving_action)

def all_observation_target_types():
    for observation in all_observations():
        for target_type in all_target_types():
            yield (observation, target_type)

def moving_traj_q_model(n_steps):
    model = {observation_moving_action: 0. for observation_moving_action in all_observation_moving_actions()}
    return model

def targetting_traj_q_model(n_steps):
    model = {observation_target_type: 0. for observation_target_type in all_observation_target_types()}
    return model

def traj_v_model(n_steps):
    model = {observation: 0. for observation in all_observations()}
    return model


def moving_stepped_q_model(n_steps):
    model = {observation_moving_action: [0.] * n_steps for observation_moving_action in all_observation_moving_actions()}
    return model

def targetting_stepped_q_model(n_steps):
    model = {observation_target_type: [0.] * n_steps for observation_target_type in all_observation_target_types()}
    return model


def stepped_v_model(n_steps):
    model = {observation: [0.] * n_steps for observation in all_observations()}
    return model


def noc(args):
    args["moving_critic"] =  None
    args["targetting_critic"] =  None

def mtc(args):
    n_steps = args["n_steps"]

    moving_model = moving_traj_q_model(n_steps)
    args["moving_critic"] = MidTrajCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = MeanTrajKalmanLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]

    targetting_model = targetting_traj_q_model(n_steps)
    args["targetting_critic"] = MidTrajCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = MeanTrajKalmanLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

#
# def msc(args):
#     n_steps = args["n_steps"]
#
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = MidSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = MidSteppedCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#
# def mhc(args):
#     n_steps = args["n_steps"]
#
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = MidHybridCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["moving_critic"].stepped_core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = MidHybridCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = MeanSteppedKalmanLearningRateScheme(args["targetting_critic"].stepped_core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]


def imtc(args):
    n_steps = args["n_steps"]

    moving_model = moving_traj_q_model(n_steps)
    args["moving_critic"] = InexactMidTrajCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]

    targetting_model = targetting_traj_q_model(n_steps)
    args["targetting_critic"] = InexactMidTrajCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

# def imtc_slow(args):
#     n_steps = args["n_steps"]
#
#     moving_model = moving_traj_q_model(n_steps)
#     args["moving_critic"] = InexactMidTrajCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#     args["moving_critic"].learning_rate = 1. / n_steps
#
#     targetting_model = targetting_traj_q_model(n_steps)
#     args["targetting_critic"] = InexactMidTrajCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#     args["targetting_critic"].learning_rate = 1. / n_steps

#
# def imsc(args):
#     n_steps = args["n_steps"]
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = InexactMidSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = InexactMidSteppedCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

def imhc(args):
    n_steps = args["n_steps"]

    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = InexactMidHybridCritic(moving_model)
    args["moving_critic"].process_noise = args["process_noise"]


    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = InexactMidHybridCritic(targetting_model)
    args["targetting_critic"].process_noise = args["process_noise"]


def qtc(args):
    n_steps = args["n_steps"]

    moving_model = moving_traj_q_model(n_steps)
    args["moving_critic"] = QTrajCritic(moving_model)
    args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
    args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    targetting_model = targetting_traj_q_model(n_steps)
    args["targetting_critic"] = QTrajCritic(targetting_model)
    args["targetting_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].core)
    args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def qtc_l(a, b):
    def qtc_l_inner(args):
        qtc(args)
        args["trace_horizon_hist"] = (a, b)

    qtc_l_inner.__name__ = f"qtc_{a:.0f}_{b}"
    return qtc_l_inner


#
#
# def qsc(args):
#     n_steps = args["n_steps"]
#
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = QSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
#     targetting_model = targetting_stepped_q_model(n_steps)
#     args["targetting_critic"] = QSteppedCritic(targetting_model)
#     args["targetting_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].core)
#     args["targetting_critic"].learning_rate_scheme.process_noise = args["process_noise"]

def qhc(args):
    n_steps = args["n_steps"]

    moving_model = moving_stepped_q_model(n_steps)
    args["moving_critic"] = QHybridCritic(moving_model)
    args["moving_critic"].process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.

    targetting_model = targetting_stepped_q_model(n_steps)
    args["targetting_critic"] = QHybridCritic(targetting_model)
    args["targetting_critic"].process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def qhc_l(a, b):
    def qhc_l_inner(args):
        qhc(args)
        args["trace_horizon_hist"] = (a, b)

    qhc_l_inner.__name__ = f"qhc_{a:.0f}_{b}"
    return qhc_l_inner

# def qhc_l(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def qhc_l_inner(args):
#         qhc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].trace_sustain = 1.
#             args["targetting_critic"].trace_sustain = 1.
#         else:
#             args["moving_critic"].trace_sustain = trace_sustain
#             args["targetting_critic"].trace_sustain = trace_sustain
#
#     qhc_l_inner.__name__ = f"qhc_{trace_horizon:.0f}"
#     return qhc_l_inner
#
# def qsc_l(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def qsc_l_inner(args):
#         qsc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].trace_sustain = 1.
#             args["targetting_critic"].trace_sustain = 1.
#         else:
#             args["moving_critic"].trace_sustain = trace_sustain
#             args["targetting_critic"].trace_sustain = trace_sustain
#
#     qsc_l_inner.__name__ = f"qsc_{trace_horizon:.0f}"
#     return qsc_l_inner

# def biqtc(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_traj_q_model(n_steps)
#     args["moving_critic"] = BiQTrajCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
# def biqsc(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = BiQSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]
#
# def biqhc(args):
#     n_steps = args["n_steps"]
#     moving_model = moving_stepped_q_model(n_steps)
#     args["moving_critic"] = BiQSteppedCritic(moving_model)
#     args["moving_critic"].learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].core)
#     args["moving_critic"].learning_rate_scheme.process_noise = args["process_noise"]

def uqtc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_traj_q_model(n_steps)
    u_moving_model = traj_v_model(n_steps)
    args["moving_critic"] = UqTrajCritic(q_moving_model, u_moving_model)
    args["moving_critic"].u_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].u_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    q_targetting_model = targetting_traj_q_model(n_steps)
    u_targetting_model = traj_v_model(n_steps)
    args["targetting_critic"] = UqTrajCritic(q_targetting_model, u_targetting_model)
    args["targetting_critic"].u_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].u_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def uqtc_l(a, b):
    def uqtc_l_inner(args):
        uqtc(args)
        args["trace_horizon_hist"] = (a, b)

    uqtc_l_inner.__name__ = f"uqtc_{a:.0f}_{b}"
    return uqtc_l_inner


# def uqsc(args):
#     n_steps = args["n_steps"]
#
#     q_moving_model = moving_stepped_q_model(n_steps)
#     u_moving_model = stepped_v_model(n_steps)
#     args["moving_critic"] = UqSteppedCritic(q_moving_model, u_moving_model)
#     args["moving_critic"].u_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].u_critic.core, True)
#     args["moving_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
#     args["moving_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
#
#     q_targetting_model = targetting_stepped_q_model(n_steps)
#     u_targetting_model = stepped_v_model(n_steps)
#     args["targetting_critic"] = UqSteppedCritic(q_targetting_model, u_targetting_model)
#     args["targetting_critic"].u_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].u_critic.core, True)
#     args["targetting_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
#     args["targetting_critic"].u_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]

def uqhc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    u_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = UqHybridCritic(q_moving_model, u_moving_model)
    args["moving_critic"].u_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.

    q_targetting_model = targetting_stepped_q_model(n_steps)
    u_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = UqHybridCritic(q_targetting_model, u_targetting_model)
    args["targetting_critic"].u_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def uqhc_l(a, b):
    def uqhc_l_inner(args):
        uqhc(args)
        args["trace_horizon_hist"] = (a, b)

    uqhc_l_inner.__name__ = f"uqhc_{a:.0f}_{b}"
    return uqhc_l_inner

def uqchc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    u_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = UqCombinedHybridCritic(q_moving_model, u_moving_model)
    args["moving_critic"].u_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].core.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    q_targetting_model = targetting_stepped_q_model(n_steps)
    u_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = UqCombinedHybridCritic(q_targetting_model, u_targetting_model)
    args["targetting_critic"].u_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].core.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def uqchc_l(a, b):
    def uqchc_l_inner(args):
        uqchc(args)
        args["trace_horizon_hist"] = (a, b)

    uqchc_l_inner.__name__ = f"uqchc_{a:.0f}_{b}"
    return uqchc_l_inner



#
# def uqhc_l(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def uqhc_l_inner(args):
#         uqhc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].u_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].u_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].u_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].u_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     uqhc_l_inner.__name__ = f"uqhc_{trace_horizon:.0f}"
#     return uqhc_l_inner
#
# def uqhc_A(args):
#     uqhc(args)
#
#     trace_sustain_25 = (25 - 1.) / 25
#     trace_sustain_50 = (50 - 1.) / 50
#
#     trace_schedule = [trace_sustain_50] * 2000 + [trace_sustain_25] * 1000
#
#     args["moving_critic"].u_critic.trace_sustain = trace_sustain_50
#     args["moving_critic"].q_critic.trace_sustain = trace_sustain_50
#
#     args["targetting_critic"].u_critic.trace_sustain = trace_sustain_50
#     args["targetting_critic"].q_critic.trace_sustain = trace_sustain_50
#
#     args["trace_schedule"] = trace_schedule
#
#
# def uqhc_B(args):
#     uqhc(args)
#
#     n = 1 / 100
#     trace_horizons = [50 / (1 + n * x) for x in range(3000)]
#
#     trace_schedule = [(trace_horizon - 1.) / (trace_horizon) for trace_horizon in trace_horizons]
#
#     args["moving_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["moving_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["targetting_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["targetting_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["trace_schedule"] = trace_schedule
#
#
# def uqhc_C(args):
#     uqhc(args)
#
#     n = 0.05
#     trace_horizons = [500 / (1 + n * x) for x in range(3000)]
#
#     trace_schedule = [(trace_horizon - 1.) / (trace_horizon) for trace_horizon in trace_horizons]
#
#     args["moving_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["moving_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["targetting_critic"].u_critic.trace_sustain = trace_schedule[0]
#     args["targetting_critic"].q_critic.trace_sustain = trace_schedule[0]
#
#     args["trace_schedule"] = trace_schedule


#
# def uqhc_l_no_quant(trace_horizon):
#     f = uqhc_l(trace_horizon)
#     f.__name__ = f"uqhc_{trace_horizon:.0f}_no_quant"
#     return f

#
# def uqsc_l(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def uqsc_l_inner(args):
#         uqsc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].u_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].u_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].u_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].u_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     uqsc_l_inner.__name__ = f"uqsc_{trace_horizon:.0f}"
#     return uqsc_l_inner




def atc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_traj_q_model(n_steps)
    v_moving_model = traj_v_model(n_steps)
    args["moving_critic"] = ATrajCritic(q_moving_model, v_moving_model)
    args["moving_critic"].v_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].v_critic.core, True)
    args["moving_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
    args["moving_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.

    q_targetting_model = targetting_traj_q_model(n_steps)
    v_targetting_model = traj_v_model(n_steps)
    args["targetting_critic"] = ATrajCritic(q_targetting_model, v_targetting_model)
    args["targetting_critic"].v_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].v_critic.core, True)
    args["targetting_critic"].q_critic.learning_rate_scheme = TrajKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
    args["targetting_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.

def atc_l(a, b):
    def atc_l_inner(args):
        atc(args)
        args["trace_horizon_hist"] = (a, b)

    atc_l_inner.__name__ = f"atc_{a:.0f}_{b}"
    return atc_l_inner

#
# def asc(args):
#     n_steps = args["n_steps"]
#
#     q_moving_model = moving_stepped_q_model(n_steps)
#     v_moving_model = stepped_v_model(n_steps)
#     args["moving_critic"] = ASteppedCritic(q_moving_model, v_moving_model)
#     args["moving_critic"].v_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].v_critic.core, True)
#     args["moving_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["moving_critic"].q_critic.core)
#     args["moving_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["moving_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]
#
#
#     q_targetting_model = targetting_stepped_q_model(n_steps)
#     v_targetting_model = stepped_v_model(n_steps)
#     args["targetting_critic"] = ASteppedCritic(q_targetting_model, v_targetting_model)
#     args["targetting_critic"].v_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].v_critic.core, True)
#     args["targetting_critic"].q_critic.learning_rate_scheme = SteppedKalmanLearningRateScheme(args["targetting_critic"].q_critic.core)
#     args["targetting_critic"].v_critic.learning_rate_scheme.process_noise = args["process_noise"]
#     args["targetting_critic"].q_critic.learning_rate_scheme.process_noise = args["process_noise"]

def ahc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    v_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = AHybridCritic(q_moving_model, v_moving_model)
    args["moving_critic"].v_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    q_targetting_model = targetting_stepped_q_model(n_steps)
    v_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = AHybridCritic(q_targetting_model, v_targetting_model)
    args["targetting_critic"].v_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def ahc_l(a, b):
    def ahc_l_inner(args):
        ahc(args)
        args["trace_horizon_hist"] = (a, b)

    ahc_l_inner.__name__ = f"ahc_{a:.0f}_{b}"
    return ahc_l_inner

def achc(args):
    n_steps = args["n_steps"]

    q_moving_model = moving_stepped_q_model(n_steps)
    v_moving_model = stepped_v_model(n_steps)
    args["moving_critic"] = ACombinedHybridCritic(q_moving_model, v_moving_model)
    args["moving_critic"].v_critic.process_noise = args["process_noise"]
    args["moving_critic"].q_critic.process_noise = args["process_noise"]
    args["moving_critic"].core.process_noise = args["process_noise"]
    args["moving_critic"].trace_sustain = 1.


    q_targetting_model = targetting_stepped_q_model(n_steps)
    v_targetting_model = stepped_v_model(n_steps)
    args["targetting_critic"] = ACombinedHybridCritic(q_targetting_model, v_targetting_model)
    args["targetting_critic"].v_critic.process_noise = args["process_noise"]
    args["targetting_critic"].q_critic.process_noise = args["process_noise"]
    args["targetting_critic"].core.process_noise = args["process_noise"]
    args["targetting_critic"].trace_sustain = 1.


def achc_l(a, b):
    def achc_l_inner(args):
        achc(args)
        args["trace_horizon_hist"] = (a, b)

    achc_l_inner.__name__ = f"achc_{a:.0f}_{b}"
    return achc_l_inner

# def ahc_l(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def ahc_l_inner(args):
#         ahc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].v_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].v_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].v_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].v_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     ahc_l_inner.__name__ = f"ahc_{trace_horizon:.0f}"
#     return ahc_l_inner

# def asc_l(trace_horizon):
#
#     trace_sustain = (trace_horizon - 1.) / trace_horizon
#
#     def asc_l_inner(args):
#         asc(args)
#         if trace_sustain == float("inf"):
#             args["moving_critic"].v_critic.trace_sustain = 1.
#             args["moving_critic"].q_critic.trace_sustain = 1.
#
#             args["targetting_critic"].v_critic.trace_sustain = 1.
#             args["targetting_critic"].q_critic.trace_sustain = 1.
#         else:
#             args["moving_critic"].v_critic.trace_sustain = trace_sustain
#             args["moving_critic"].q_critic.trace_sustain = trace_sustain
#
#             args["targetting_critic"].v_critic.trace_sustain = trace_sustain
#             args["targetting_critic"].q_critic.trace_sustain = trace_sustain
#
#     asc_l_inner.__name__ = f"asc_{trace_horizon:.0f}"
#     return asc_l_inner
