# rl_parsers

This repository contains parsers for file formats related to reinforcement
learning.  In each case, the contents of the parsed file is returned as
a `namedtuple` containing the fields specified by the respective file format.

## POMDP:  Partially Observable Markov Decision Process

Standard POMDP file format, with the addition of the `reset` keyword, which may
be used both to indicate the end of an episode in episodic tasks, and the
reinitialization of the state according to the initial distribution in
continuiung tasks.

## MDP:  Markov Decision Process

Standard MDP file format, with the addition of the `reset` keyword, which may
be used both to indicate the end of an episode in episodic tasks, and the
reinitialization of the state according to the initial distribution in
continuiung tasks.

## FSC:  Finite State Controller

A finite state controller (FSC) is a graph-based policy used for partially
observable environments.  This is my own custom file format.

## FSS:  Finite State Structure

A finite state structure (FSS) is a format used to constraint either the
internal-dynamics or the action-selection of a FSC.  This is my own custom file
format.
