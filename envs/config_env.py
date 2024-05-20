#!/usr/bin/env python
# coding=utf8


def config_env(_flags):
    flags = _flags

    # Scenario
    flags.DEFINE_string("scenario", "predator_prey_obs", "Scenario")
    flags.DEFINE_integer("n_predator", 4, "Number of predators")
    flags.DEFINE_integer("n_prey", 4, "Number of preys")
    flags.DEFINE_boolean("obs_diagonal", True, "Whether the agent can see in diagonal directions")
    flags.DEFINE_boolean("moving_prey", False, "Whether the prey is moving")
    flags.DEFINE_integer("obs_range", 1, "Observation range")
    flags.DEFINE_integer("hetero", 1, "Heterogeneity of observation range")

    # Observation
    flags.DEFINE_integer("history_len", 1, "How many previous steps we look back")

    # core
    flags.DEFINE_integer("map_size", 12, "Size of the map")
    flags.DEFINE_float("render_every", 1, "Render the nth episode")

    # GUI
    flags.DEFINE_boolean("gui", True, "Activate GUI")


def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "s-"+FLAGS.scenario+"-map-"+str(FLAGS.map_size)+"-or-"+str(FLAGS.obs_range)
