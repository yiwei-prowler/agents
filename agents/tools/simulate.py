# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-graph simulation step of a vecrotized algorithm with environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from agents.tools import streaming_mean


def simulate(batch_env, algo, should_step, use_external_action, external_action, log=True, reset=False):
  """Simulation a single step of a vecrotized algorithm with in-graph environments.
  This means perform an action on all environments (environments that are not-started or
  terminated are re-started to make sure that all environments are always available).

  Integrates the operations implemented by the algorithm and the environments
  into a combined operation.

  Args:
    batch_env: In-graph batch environment.
    algo: Algorithm instance implementing required operations.
    log: Tensor indicating whether to compute and return summaries.
    reset: Tensor causing all environments to reset.

  Returns:
    Tuple of tensors containing done flags for the current episodes, possibly
    intermediate scores for the episodes, and a summary tensor.
  """

  def _define_begin_episode(agent_indices):
    """Reset environments, intermediate scores and durations for new episodes.

    Args:
      agent_indices: Tensor containing batch indices starting an episode.

    Returns:
      Summary tensor.
    """
    assert agent_indices.shape.ndims == 1
    zero_scores = tf.zeros_like(agent_indices, tf.float32)
    zero_durations = tf.zeros_like(agent_indices)

    reset_ops = [
        batch_env.reset(agent_indices),
        tf.scatter_update(score, agent_indices, zero_scores),
        tf.scatter_update(length, agent_indices, zero_durations)]

    with tf.control_dependencies(reset_ops):
      return algo.begin_episode(agent_indices)

  def _define_step():
    """Request actions from the algorithm and apply them to the environments.

    Increments the lengths of all episodes and increases their scores by the
    current reward. After stepping the environments, provides the full
    transition tuple to the algorithm.

    Returns:
      Summary tensor.
    """
    # prevob is the list of observations, one per each environment.
    prevob = batch_env.observ + 0  # Ensure a copy of the variable value.
    # Perform actions given observations on ALL environments.

    def _action_from_algo():
        action, step_summary = algo.perform(prevob)
        x = tf.Print(action, [action], 'This is internal action:')
        with tf.control_dependencies([x]):
            return action, step_summary

    def _action_from_external():
        x = tf.Print(external_action, [external_action], 'This is external action:')
        with tf.control_dependencies([x]):
            return external_action, tf.constant('')

    action, step_summary = tf.cond(use_external_action, _action_from_external, _action_from_algo)
    with tf.control_dependencies([action, step_summary]):
        # action, step_summary = algo.perform(prevob)
        action.set_shape(batch_env.action.shape)
        with tf.control_dependencies([batch_env.simulate(action)]):
          add_score = score.assign_add(batch_env.reward)
          inc_length = length.assign_add(tf.ones(len(batch_env), tf.int32))
        with tf.control_dependencies([add_score, inc_length]):
          experience_summary = algo.experience(
              prevob, batch_env.action, batch_env.reward, batch_env.done,
              batch_env.observ)
        return tf.summary.merge([step_summary, experience_summary])

  def _define_end_episode(agent_indices):
    """Notify the algorithm of ending episodes.

    Also updates the mean score and length counters used for summaries.

    Args:
      agent_indices: Tensor holding batch indices that end their episodes.

    Returns:
      Summary tensor.
    """
    assert agent_indices.shape.ndims == 1
    submit_score = mean_score.submit(tf.gather(score, agent_indices))
    submit_length = mean_length.submit(
        tf.cast(tf.gather(length, agent_indices), tf.float32))
    with tf.control_dependencies([submit_score, submit_length]):
      return algo.end_episode(agent_indices)

  def _define_summaries():
    """Reset the average score and duration, and return them as summary.

    Returns:
      Summary string.
    """
    score_summary = tf.cond(
        tf.logical_and(log, tf.cast(mean_score.count, tf.bool)),
        lambda: tf.summary.scalar('mean_score', mean_score.clear()), str)
    length_summary = tf.cond(
        tf.logical_and(log, tf.cast(mean_length.count, tf.bool)),
        lambda: tf.summary.scalar('mean_length', mean_length.clear()), str)
    return tf.summary.merge([score_summary, length_summary])

  def _perform_step():
      step = _define_step()
      # Find out which environments are done, and call end_episode on agents
      # for those done environments. end_episode will trigger training if we are training
      # and there is enough episode generated. Different environments may terminated at
      # different time steps. Here we collect only the indices for environments that are terminated
      # after the previous step. The triggering is in ppo.algorithm.PPOAlgorithm._define_end_episode.
      # What does enough episode mean? In PPOAlgorithm, the _episodes tensor stores the episodes
      # that is happening during simulation in environments. Once an environment has terminated,
      # its episode is copied to the _memory tensor. When the size of the _memory tensor reaches
      # config.update_every, it means there is enough training episodes to for the next batch
      # of training, so training starts.
      with tf.control_dependencies([step]):
          agent_indices = tf.cast(tf.where(batch_env.done)[:, 0], tf.int32)
          end_episode = tf.cond(
              tf.cast(tf.shape(agent_indices)[0], tf.bool),
              lambda: _define_end_episode(agent_indices), str)

      with tf.control_dependencies([end_episode]):
          summary = tf.summary.merge([
              _define_summaries(), begin_episode, step, end_episode])
      with tf.control_dependencies([summary]):
          done, score2 = tf.identity(batch_env.done), tf.identity(score)
      return done, score2, summary

  def _donot_perform_step():
      summary = tf.summary.merge([begin_episode])
      with tf.control_dependencies([summary]):
          done, score2 = tf.identity(batch_env.done), tf.identity(score)
      return done, score2, summary

  # Simulate a single step for all the environments.
  # Restart environments that are done if necessary.

  with tf.name_scope('simulate'):
    log = tf.convert_to_tensor(log)
    reset = tf.convert_to_tensor(reset)
    with tf.variable_scope('simulate_temporary'):

      score = tf.Variable(tf.zeros(len(batch_env), dtype=tf.float32), False, name='score')
      length = tf.Variable(tf.zeros(len(batch_env), dtype=tf.int32), False, name='length')

    mean_score = streaming_mean.StreamingMean((), tf.float32)
    mean_length = streaming_mean.StreamingMean((), tf.float32)

    # Find out the indices of environments which are done. or in the case reset is True,
    # all the indices of environments.
    agent_indices = tf.cond(
        reset,
        lambda: tf.range(len(batch_env)),
        lambda: tf.cast(tf.where(batch_env.done)[:, 0], tf.int32))

    # If some environments (given by agent_indices) are done,
    # restart those environments.
    begin_episode = tf.cond(
        tf.cast(tf.shape(agent_indices)[0], tf.bool),
        lambda: _define_begin_episode(agent_indices), str)

    # Simulate one step in all environments.
    with tf.control_dependencies([begin_episode]):
      return tf.cond(should_step, _perform_step, _donot_perform_step)

    #   step = _define_step()
    #
    # # Find out which environments are done, and call end_episode on agents
    # # for those done environments. end_episode will trigger training if we are training
    # # and there is enough episode generated. Different environments may terminated at
    # # different time steps. Here we collect only the indices for environments that are terminated
    # # after the previous step. The triggering is in ppo.algorithm.PPOAlgorithm._define_end_episode.
    # # What does enough episode mean? In PPOAlgorithm, the _episodes tensor stores the episodes
    # # that is happening during simulation in environments. Once an environment has terminated,
    # # its episode is copied to the _memory tensor. When the size of the _memory tensor reaches
    # # config.update_every, it means there is enough training episodes to for the next batch
    # # of training, so training starts.
    # with tf.control_dependencies([step]):
    #   agent_indices = tf.cast(tf.where(batch_env.done)[:, 0], tf.int32)
    #   end_episode = tf.cond(
    #       tf.cast(tf.shape(agent_indices)[0], tf.bool),
    #       lambda: _define_end_episode(agent_indices), str)
    #
    # with tf.control_dependencies([end_episode]):
    #   summary = tf.summary.merge([
    #       _define_summaries(), begin_episode, step, end_episode])
    # with tf.control_dependencies([summary]):
    #   done, score = tf.identity(batch_env.done), tf.identity(score)
    # return done, score, summary

    # else:
    #     with tf.control_dependencies([begin_episode]):
    #         summary = tf.summary.merge([begin_episode])
    #         with tf.control_dependencies([summary]):
    #             done, score = tf.identity(batch_env.done), tf.identity(score)
    #         return done, score, summary
