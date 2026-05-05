DOMAINS = [
    'walker',
    'hopper',
    'cheetah',
    'quadruped',
    'humanoid',
    'jaco',
    'ant',
]

HOPPER_TASKS = [
    'hopper_hop',
    'hopper_stand'
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

CHEETAH_TASKS = [
    'cheetah_run',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

ANT_TASKS = [
    'ant_temp',
]

ANTMAZE_TASKS = [
    'antmaze_umaze',
    'antmaze_medium',
    'antmaze_medium_diverse',
    'antmaze_large',
    'antmaze_large_diverse',
]

HUMANOID_TASKS = [
    'humanoid_stand',
    'humanoid_walk',
    'humanoid_run',
    'humanoid-run_pure_state-v0',
]

TASKS = WALKER_TASKS + HOPPER_TASKS + CHEETAH_TASKS + QUADRUPED_TASKS + JACO_TASKS + ANT_TASKS + ANTMAZE_TASKS + HUMANOID_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'hopper': 'hopper_hop',
    'cheetah': 'cheetah_run',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'ant': 'ant_temp',
    'antmaze': 'antmaze_medium',
    'humanoid': 'humanoid_walk',
}
