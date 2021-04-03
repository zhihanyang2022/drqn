from functools import lru_cache

import numpy as np

__all__ = ['belief_init', 'belief_step', 'expected_reward', 'expected_obs']


def belief_init(env, shape=None) -> np.array:
    """Return batch of initial belief-states.

    :param env:  Gym-POMDPs environment
    :param shape:  Batch shape of belief-states
    :rtype:  numpy.ndarray (*shape, |S|) batch array of belief-states
    """

    if shape is None:
        shape = ()

    return np.tile(env.start, shape + (1,))


def belief_step(env, b, a, o) -> np.array:
    """Return batch of updated belief-states.

    :param env:  Gym-POMDPs environment
    :param b:  (*, |S|) batch array of belief-states
    :param a:  (*,) batch array of actions
    :param o:  (*,) batch array of observations
    :rtype:  numpy.array (*, |S|) batch array of next belief-states
    """
    b = np.asarray(b)
    a = np.asarray(a)
    o = np.asarray(o)

    if not b.shape[:-1] == a.shape == o.shape:
        raise ValueError('Input array shapes do not match')

    bshape = b.shape
    b = b.reshape(-1, env.state_space.n)
    a = a.reshape(-1)
    o = o.reshape(-1)

    if not _plausible(env, b, a, o).all():
        raise ValueError('impossible observation from given belief-action pair')

    p_sa_so = _P_SA_SO(env)[:, a, :, o]
    b1 = np.einsum('...ij,...i->...j', p_sa_so, b)
    b1 = b1 / b1.sum(-1, keepdims=True)
    b1 = b1.reshape(bshape)
    return b1


def expected_reward(env, b, a) -> np.ndarray:
    """Compute R(b, a) = E_{s\\sim b}[ R(s, a) ]

    :param env:  Gym-POMDPs environment
    :param b:  (*, |S|) batch array of belief-states
    :param a:  (*,) batch array of actions
    :rtype: np.ndarray  (*,) batch array of expected rewards
    """
    b = np.asarray(b)
    a = np.asarray(a)

    if not b.shape[:-1] == a.shape:
        raise ValueError('Input array shapes do not match')

    ashape = a.shape
    b = b.reshape(-1, env.state_space.n)
    a = a.reshape(-1)

    e_sa_r = _E_SA_R(env)[:, a]
    expected_r = np.einsum('ib,bi->b', e_sa_r, b)
    return expected_r.reshape(ashape)


def expected_obs(env, b, a) -> np.ndarray:
    """Computes Pr(o \\mid b, a) = E_{s\\sim b}[ p(o|s, a) ]

    :param env:  Gym-POMDPs environment
    :param b:  (*, |S|) batch array of belief states
    :param a:  (*,) batch array of actions
    :rtype: np.ndarray  (*, O) batch array of observation probabilities
    """
    b = np.asarray(b)
    a = np.asarray(a)

    if not b.shape[:-1] == a.shape:
        raise ValueError('Input array shapes do not match')

    ashape = a.shape
    b = b.reshape(-1, env.state_space.n)
    a = a.reshape(-1)

    p_sa_o = _P_SA_O(env)[:, a, :]
    eobs = np.einsum('ibj,bi->bj', p_sa_o, b)
    return eobs.reshape(*ashape, -1)


def _plausible(env, b, a, o) -> np.ndarray:
    """Return plausibility of observations following belief-action pair.

    :param env:  Gym-POMDPs environment
    :param b:  (*, |S|) batch array of belief states
    :param a:  (*,) batch array of actions
    :param o:  (*,) batch array of observations
    :rtype: np.ndarray  (*,) batch array of bools
    """
    p_sa_o = _P_SA_O(env)[:, a, o]
    return np.einsum('ib,bi->b', p_sa_o, b) > 0.0


@lru_cache(maxsize=None)
def _P_SA_SO(env) -> np.ndarray:
    """Compute the generative matrix G_{ijkl} = \\Pr(s'=k, o=l \\mid s=i, a=j)

    :param env:  Gym-POMDPs environment
    :rtype: np.ndarray  (|S|, |A|, |S|, |O|) batch array
    """
    return np.expand_dims(env.T, axis=-1) * env.O


@lru_cache(maxsize=None)
def _E_SA_R(env) -> np.ndarray:
    """Compute the expected rewards matrix R_{ij} = E[r \\mid s=i, a=j]

    :param env:  Gym-POMDPs environment
    :rtype: np.ndarray  (|S|, |A|) batch array of expected rewards
    """
    return (_P_SA_SO(env) * env.R).sum((-2, -1))


@lru_cache(maxsize=None)
def _P_SA_O(env) -> np.ndarray:
    """Compute the observation matrix O_{ijl} = \\Pr(o=l \\mid s=i, a=j)

    :param env:  Gym-POMDPs environment
    :rtype: np.ndarray  (|S|, |A|, |O|) batch array of observation probabilities
    """
    return _P_SA_SO(env).sum(-2)
