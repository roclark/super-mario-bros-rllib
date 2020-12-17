import gym
import gym_super_mario_bros
import ray
from argparse import ArgumentParser
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.env.atari_wrappers import (MonitorEnv,
                                          NoopResetEnv,
                                          WarpFrame,
                                          FrameStack)
from tabulate import tabulate


# An updated version of the EpisodicLifeEnv wrapper from RLLib which is
# compatible with the SuperMarioBros environments.
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over. Done by DeepMind for the DQN and co. since it helps value
        estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info


def parse_args():
    parser = ArgumentParser(description='Train an agent to beat Super Mario '
                            'Bros. levels.')
    parser.add_argument('--checkpoint', help='Specify an existing checkpoint '
                        'which can be used to restore progress from a previous'
                        ' training run.')
    parser.add_argument('--dimension', help='The image dimensions to resize to'
                        ' while preprocessing the game states.', type=int,
                        default=84)
    parser.add_argument('--environment', help='The Super Mario Bros level to '
                        'train on.', type=str, default='SuperMarioBros-1-1-v0')
    parser.add_argument('--framestack', help='The number of frames to stack '
                        'together to feed into the network.', type=int,
                        default=4)
    parser.add_argument('--gpus', help='Number of GPUs to include in the '
                        'cluster.', type=int, default=0)
    parser.add_argument('--iterations', help='Number of iterations to train '
                        'for.', type=int, default=1000000)
    parser.add_argument('--workers', help='Number of workers to launch on the '
                        'cluster. Hint: Must be less than the number of CPU '
                        'cores available.', type=int, default=4)
    return parser.parse_args()


def env_creator(env_name, config, dim, framestack):
    env = gym_super_mario_bros.make(env_name)
    env = CustomReward(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env, dim)
    if framestack:
        env = FrameStack(env, framestack)
    return env


def print_results(result, iteration):
    table = [['IMPALA',
              iteration,
              result['timesteps_total'],
              round(result['episode_reward_max'], 3),
              round(result['episode_reward_min'], 3),
              round(result['episode_reward_mean'], 3)]]
    print(tabulate(table,
                   headers=['Agent',
                            'Iteration',
                            'Steps',
                            'Max Reward',
                            'Min Reward',
                            'Mean Reward'],
                   tablefmt='psql',
                   showindex="never"))
    print()


def main():
    def env_creator_lambda(env_config):
        return env_creator(args.environment,
                           config,
                           args.dimension,
                           args.framestack)

    args = parse_args()
    config = {
        'env': 'super_mario_bros',
        'framework': 'torch',
        'rollout_fragment_length': 50,
        'train_batch_size': 500,
        'num_workers': args.workers,
        'num_envs_per_worker': 1,
        'num_gpus': args.gpus
    }
    ray.init()

    register_env('super_mario_bros', env_creator_lambda)
    trainer = ImpalaTrainer(config=config)

    if args.checkpoint:
        trainer.restore(args.checkpoint)

    for iteration in range(args.iterations):
        result = trainer.train()
        print_results(result, iteration)

        if iteration % 50 == 0:
            checkpoint = trainer.save()
            print('Checkpoint saved at', checkpoint)


if __name__ == "__main__":
    main()
