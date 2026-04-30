DOMAINS = [
    'walker',
    'hopper',
    'quadruped',
    'humanoid',
    'jaco',
    'antmaze',
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

ANTMAZE_TASKS = [
    'antmaze_umaze',
    'antmaze_medium_play',
    'antmaze_medium_diverse',
    'antmaze_large_play',
    'antmaze_large_diverse',
]

HUMANOID_TASKS = [
    'humanoid_stand',
    'humanoid_walk',
    'humanoid_run',
    'humanoid-run_pure_state-v0',
]

TASKS = WALKER_TASKS + HOPPER_TASKS + QUADRUPED_TASKS + JACO_TASKS + ANTMAZE_TASKS + HUMANOID_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'hopper': 'hopper_hop',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'antmaze': 'antmaze_umaze',
    'humanoid': 'humanoid_walk',
}
