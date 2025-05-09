import gym
import numpy as np
from utils.Memory import ExperienceReplay
from utils.logger_utils import get_logger
from model import CartPoleSAC                 # ← 改用 SAC
from configs import CartpoleConfig as cfg     # 同一份設定檔，可在其中新增 lr、tau 等 SAC 參數

# ------------------------------------------------------------
def sample_batch(exp_replay, batch_size):
    """從 ExperienceReplay 抽樣並整理成 SAC 需要的字典格式"""
    batch = exp_replay.sample(batch_size)
    states      = np.array([item[0] for item in batch])
    actions     = np.array([[item[1]] for item in batch], dtype=np.int32)
    rewards     = np.array([[item[2]] for item in batch], dtype=np.float32)
    next_states = np.array([item[3] for item in batch])
    dones       = np.array([[float(item[4])] for item in batch], dtype=np.float32)
    return {'s': states,
            'a': actions,
            'r': rewards,
            's_': next_states,
            'd': dones}

# ------------------------------------------------------------
def main():
    agent = CartPoleSAC(input_size=cfg.input_size,
                        output_size=cfg.output_size,
                        model_path=cfg.model_path_sac)
    agent.init_target()                       # 初始化 target critics
    logger = get_logger("train_Cartpole_SAC")
    fit(logger, agent, cfg.n_epoch)

# ------------------------------------------------------------
def fit(logger, agent: CartPoleSAC, n_epochs=1500):
    logger.info("------Building env------")
    env = gym.make('CartPole-v0')
    last_mean_100_reward = [0] * 100
    exp_replay = ExperienceReplay(size=cfg.memory_size)
    logger.info("------Commence training------")

    total_steps = 0
    i = 0  # for convergence purposes
    for e in range(n_epochs):
        state = env.reset()
        done = False
        epoch_reward = 0

        while not done:
            total_steps += 1
            action = agent.sample_action(np.expand_dims(state, 0))[0]
            next_state, reward, done, _ = env.step(action)
            epoch_reward += reward
            exp_replay.add_memory(state, action, reward, next_state, done)
            state = next_state

            if total_steps < cfg.OBSERVE or exp_replay.size < cfg.batch_size:
                continue

            batch_dict = sample_batch(exp_replay, cfg.batch_size)
            agent.learn(batch_dict, lr=cfg.lr)

        # 記錄與評估
        last_mean_100_reward[e % 100] = epoch_reward
        if e < 100:
            print("Episode ", e, " / {} finished with reward {}".format(n_epochs, epoch_reward))
        else:
            mean_100_reward = sum(last_mean_100_reward) / 100
            print("Episode ", e,
                  " / {} finished with reward {} | last‑100 avg {}".format(n_epochs,
                                                                           epoch_reward,
                                                                           mean_100_reward))
            logger.info("Episode {} / {} reward {} | last‑100 avg {}".format(e,
                                                                             n_epochs,
                                                                             epoch_reward,
                                                                             mean_100_reward))
            if mean_100_reward > cfg.OBJECTIVE_SCORE:
                agent.save_model()
                logger.info("Goal achieved! ep {}‑{} avg reward {}".format(e - 100, e, mean_100_reward))
                i += 1
                if i % 50 == 0:
                    break
            else:
                i = 0
    try:
        del env
    except ImportError:
        pass

# ------------------------------------------------------------
if __name__ == '__main__':
    main()
