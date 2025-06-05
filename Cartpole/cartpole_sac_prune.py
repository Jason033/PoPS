# -*- coding: utf-8 -*-
"""
SAC Model Pruning for CartPole Environment
Based on PoPS iterative pruning methodology adapted for Soft Actor-Critic
"""

import gym
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import os

# Import from the PoPS framework
from configs import CartpoleConfig as base_config
from utils.Memory import ExperienceReplay, Supervised_ExperienceReplay, Supervised_Prioritzed_ExperienceReplay
from utils.logger_utils import get_logger
from utils.plot_utils import plot_graph
from model import CartPoleSAC
from Cartpole.evaluate_cartpole import evaluate_cartepole
import sys
#sys.path.append('../PoPS-main')


class PruneCartpoleSACConfig:
    """Configuration for SAC CartPole Pruning"""
    # Environment and model
    input_size = (None, 4)
    output_size = (None, 2)
    
    # Paths
    model_path_sac = 'save_model/SAC_model'
    prune_best_actor = 'saved_model/Cart_pole_SAC/best_prune_sac'
    iterative_PoPS = 'PoPS_iterative_SAC'
    
    # Training parameters
    n_epoch = 100
    n_pruning_iterations = 50
    batch_size = 256
    memory_size = 100000
    OBSERVE_PERIOD_EXPERIENCE = 2000
    
    # SAC hyperparameters
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    gamma = 0.99
    tau = 0.005
    
    # Performance targets
    objective_score = 195.0
    lower_bound_score = 150.0
    eval_episodes = 10
    
    # Prioritized Experience Replay
    ALPHA_PER = 0.6
    EPS_PER = 1e-6
    BETA0_PER = 0.4
    
    # Pruning parameters
    pruning_freq = 10
    sparsity_end = int(5e5)
    target_sparsity = 0.99
    tau_target_update = 0.005
    
    @staticmethod
    def learning_rate_schedule_actor(iteration, base_lr_actor):
        """Dynamic learning rate schedule for actor network"""
        if iteration <= 10:
            return base_lr_actor
        elif iteration <= 20:
            return base_lr_actor * 0.5
        elif iteration <= 30:
            return base_lr_actor * 0.1
        else:
            return base_lr_actor * 0.05
    
    @staticmethod
    def learning_rate_schedule_critic(iteration, base_lr_critic):
        """Dynamic learning rate schedule for critic networks"""
        if iteration <= 10:
            return base_lr_critic
        elif iteration <= 20:
            return base_lr_critic * 0.5
        elif iteration <= 30:
            return base_lr_critic * 0.1
        else:
            return base_lr_critic * 0.05
    
    @staticmethod
    def beta_schedule(beta0, e, n_epoch):
        """Beta schedule for Prioritized Experience Replay"""
        return min(beta0 + ((1 - beta0) / n_epoch) * e, 1.0)


class StudentCartPoleSAC(CartPoleSAC):
    """Student SAC model with pruning capabilities"""
    
    def __init__(self, input_size, output_size, model_path,
                 scope='StudentCartPoleSAC', gamma=0.99, tau=0.005,
                 alpha_init=0.2, target_entropy=None,
                 pruning_start=0, pruning_end=-1, pruning_freq=10,
                 sparsity_start=0, sparsity_end=int(10e5),
                 target_sparsity=0.9, initial_sparsity=0):
        
        # Pruning parameters
        self.pruning_start = pruning_start
        self.pruning_end = pruning_end
        self.pruning_freq = pruning_freq
        self.sparsity_start = sparsity_start
        self.sparsity_end = sparsity_end
        self.target_sparsity = target_sparsity
        self.initial_sparsity = initial_sparsity
        self.global_step = 0
        self.frozen_global_step = None
        self.scope = scope
        # Initialize parent class
        super(StudentCartPoleSAC, self).__init__(
            input_size=input_size, 
            output_size=output_size, 
            model_path=model_path,
            scope=scope, 
            gamma=gamma, 
            tau=tau,
            alpha_init=alpha_init, 
            target_entropy=target_entropy
        )
        
        # Add pruning operations
        self._build_pruning_ops()
        # Initialize the mask variables created by _build_pruning_ops
        # These masks are specific to the Student model and its pruning mechanism.
        if hasattr(self, 'masks') and self.masks:
            # Ensure the session is available (it should be from super().__init__ via BaseNetwork)
            _ = self.sess # Accesses the property to ensure _sess is created if not already
            self.sess.run(tf.compat.v1.variables_initializer(self.masks))
    
    def _build_pruning_ops(self):
        """Build pruning operations for magnitude-based pruning"""
        with self.graph.as_default():
            # Get all trainable variables
            trainable_vars = tf.trainable_variables()
            
            # Filter for weight matrices only (exclude biases)
            self.weight_vars = [var for var in trainable_vars
                               if 'kernel' in var.name or 'weight' in var.name]
            
            # Create masks for each weight variable
            self.masks = []
            self.mask_assigns = []
            self.sparsity_ops = []
            
            print("=== [Debug] _build_pruning_ops: weight_vars ===")
            for idx, var in enumerate(self.weight_vars):
                print(f"  weight_vars[{idx}]: {var.name}, shape: {var.shape}")

            for idx, var in enumerate(self.weight_vars):
                # Create mask variable
                mask = tf.Variable(
                    tf.ones_like(var),
                    trainable=False,
                    name=var.name.replace(':', '_') + '_mask'
                )
                self.masks.append(mask)
                
                # Magnitude-based pruning
                threshold = tf.contrib.distributions.percentile(
                    tf.abs(var),
                    self.target_sparsity * 100
                )
                new_mask = tf.cast(tf.abs(var) >= threshold, tf.float32)
                
                # Mask assignment operation
                mask_assign = mask.assign(new_mask)
                self.mask_assigns.append(mask_assign)
                
                # Apply mask to variable
                masked_var = var * mask
                sparsity_op = tf.assign(var, masked_var)
                self.sparsity_ops.append(sparsity_op)
                
                # Calculate sparsity
                sparsity = 1.0 - tf.reduce_mean(mask)
                self.sparsity_ops.append(sparsity)
                
    
    def prune(self):
        """Perform magnitude-based pruning"""
        if self.frozen_global_step is not None:
            return  # Skip pruning if frozen
            
        self.global_step += 1
        
        # Check if it's time to prune
        if (self.global_step >= self.pruning_start and 
            (self.pruning_end == -1 or self.global_step <= self.pruning_end) and
            self.global_step % self.pruning_freq == 0):
            
            # Update masks and apply pruning
            self.sess.run(self.mask_assigns)
            self.sess.run(self.sparsity_ops[:-len(self.weight_vars)])  # Exclude sparsity calculations
    
    def get_model_sparsity(self):
        """Get current model sparsity"""
        if not hasattr(self, 'sparsity_ops') or len(self.sparsity_ops) == 0:
            return 0.0
        
        sparsities = []
        # 只取 sparsity tensor (每個 var 的第二個 op, 即奇數 index)
        for i, op in enumerate(self.sparsity_ops[1::2]):
            sparsity = self.sess.run(op)
            if hasattr(sparsity, 'shape'):
                print(f"    [Debug] sess.run(sparsity_ops[{2*i+1}]) type: {type(sparsity)}, shape: {sparsity.shape}")
            else:
                print(f"    [Debug] sess.run(sparsity_ops[{2*i+1}]) type: {type(sparsity)}")
            sparsities.append(sparsity)
        
        return np.mean(sparsities)
    
    def get_number_of_nnz_params(self):
        """Get number of non-zero parameters"""
        total_params = 0
        nnz_params = 0
        
        for var in self.weight_vars:
            var_val = self.sess.run(var)
            total_params += np.prod(var_val.shape)
            nnz_params += np.count_nonzero(var_val)
        
        return nnz_params
    
    def get_number_of_nnz_params_per_layer(self):
        """Get number of non-zero parameters per layer"""
        nnz_per_layer = []
        
        for var in self.weight_vars:
            var_val = self.sess.run(var)
            nnz_per_layer.append(np.count_nonzero(var_val))
        
        return nnz_per_layer
    
    def freeze_global_step(self):
        """Freeze global step to stop pruning"""
        self.frozen_global_step = self.global_step
        return self.frozen_global_step
    
    def unfreeze_global_step(self):
        """Unfreeze global step to resume pruning"""
        frozen_step = self.frozen_global_step
        self.frozen_global_step = None
        return frozen_step
    
    def reset_global_step(self):
        """Reset global step"""
        self.global_step = 0
        self.frozen_global_step = None


def sample_batch_sac(exp_replay, batch_size):
    """Sample batch from experience replay for SAC training"""
    batch = exp_replay.sample(batch_size)
    states = np.stack([item[0] for item in batch], axis=0).astype(np.float32)
    actions = np.array([[item[1]] for item in batch], dtype=np.int32)
    rewards = np.array([[item[2]] for item in batch], dtype=np.float32)
    next_states = np.stack([item[4] for item in batch], axis=0).astype(np.float32)
    dones = np.array([np.squeeze(item[3]).astype(np.float32) for item in batch], 
                    dtype=np.float32).reshape(-1, 1)
    
    return {
        's': states,
        'a': actions,
        'r': rewards,
        's_': next_states,
        'd': dones
    }


def accumulate_experience_sac_cartpole(teacher_agent, exp_replay, num_steps, env):
    """Accumulate experience using teacher agent for student training"""
    steps = 0
    
    while steps < num_steps:
        state = env.reset()
        done = False
        
        while not done and steps < num_steps:
            steps += 1
            action = teacher_agent.sample_action(np.expand_dims(state, 0), explore=False)[0]
            next_state, reward, done, _ = env.step(action)
            exp_replay.add_memory(state, action, reward, next_state, done)
            state = next_state


def evaluate_sac_cartpole(agent, n_epoch=10, render=False):
    """Evaluate SAC agent on CartPole environment"""
    env = gym.make('CartPole-v0')
    mean_reward = np.zeros(n_epoch)
    
    for e in range(n_epoch):
        state = env.reset()
        done = False
        epoch_reward = 0
        
        while not done:
            if render:
                env.render()
            action = agent.sample_action(np.expand_dims(state, 0), explore=False)[0]
            next_state, reward, done, _ = env.step(action)
            epoch_reward += reward
            state = next_state
            
        mean_reward[e] = epoch_reward
    
    mean_reward = np.mean(mean_reward)
    print(f"Evaluation over {n_epoch} episodes: {mean_reward}")
    return mean_reward


def train_sac_student_cartpole(logger, student_agent, exp_replay, env,
                              prune_this_iteration=False,
                              best_model_path=None,
                              lr_actor=1e-4, lr_critic=1e-4,
                              stop_pruning_temporarily=True,
                              use_per=False,
                              current_pruning_iteration=0,
                              num_train_steps=20000,
                              objective_score=195.0,
                              lower_bound_score=150.0,
                              config=PruneCartpoleSACConfig):
    """Train SAC student with optional pruning"""
    
    nnz_params_history, score_history = [], []
    stop_pruning_flag = stop_pruning_temporarily
    low_score_count = 0
    
    logger.info(f"SAC student training | LR_actor={lr_actor:.1e}, "
                f"LR_critic={lr_critic:.1e}, Prune={prune_this_iteration}")
    
    last_sparsity_measure = -1
    
    for step in range(num_train_steps + 1):
        if prune_this_iteration and not stop_pruning_flag:
            student_agent.prune()
        
        # Sample batch and train
        if exp_replay.size >= config.batch_size:
            if use_per:
                # Use prioritized experience replay (simplified)
                batch_dict = sample_batch_sac(exp_replay, config.batch_size)
            else:
                batch_dict = sample_batch_sac(exp_replay, config.batch_size)
            
            # Train with current learning rates
            actor_loss, critic_loss, alpha_loss, alpha_val = student_agent.learn(
                batch_dict, lr=lr_actor
            )
        
        # Evaluation and logging
        if step % 1000 == 0:
            if prune_this_iteration:
                score = evaluate_sac_cartpole(student_agent, n_epoch=config.eval_episodes)
                sparsity = student_agent.get_model_sparsity()
                
                logger.info(f"Step {step}/{num_train_steps}: sparsity={sparsity:.4f}, "
                           f"{config.eval_episodes}-episode mean score={score:.2f}")
                
                if sparsity > last_sparsity_measure:
                    nnz = student_agent.get_number_of_nnz_params()
                    score_history.append(score)
                    nnz_params_history.append(nnz)
                    last_sparsity_measure = sparsity
                elif len(score_history) > 0 and score > score_history[-1]:
                    score_history[-1] = score
                
                # Check if agent achieved objective
                if score > objective_score:
                    if stop_pruning_flag:
                        freeze_step = student_agent.unfreeze_global_step()
                        logger.info(f"Agent recovered, resuming pruning at step {freeze_step}")
                        stop_pruning_flag = False
                    
                    if best_model_path:
                        logger.info(f"Saving best model with sparsity {sparsity:.4f} to {best_model_path}")
                        student_agent.save_model(path=best_model_path)
                
                # Check if performance dropped too low
                if score < lower_bound_score and not stop_pruning_flag:
                    stop_pruning_flag = True
                    freeze_step = student_agent.freeze_global_step()
                    logger.info(f"Performance dropped, stopping pruning at step {freeze_step}")
                
                if score < lower_bound_score:
                    low_score_count += 1
                    if low_score_count >= 5:
                        logger.info("Breaking due to consistently low performance")
                        break
                else:
                    low_score_count = 0
    
    return score_history, nnz_params_history, stop_pruning_flag


def iterative_pruning_sac_cartpole(logger, student_agent, env,
                                  iterations=50, use_per=False,
                                  config=PruneCartpoleSACConfig,
                                  best_actor_path=None,
                                  objective_score=195.0,
                                  lower_bound_score=150.0):
    """Iterative pruning procedure for SAC CartPole agent"""
    
    initial_score = evaluate_sac_cartpole(student_agent, n_epoch=config.eval_episodes)
    nnz_vs_accuracy = [[], []]
    
    initial_nnz = student_agent.get_number_of_nnz_params()
    nnz_vs_accuracy[0].append(initial_nnz)
    nnz_vs_accuracy[1].append(initial_score)
    
    logger.info(f"Initial evaluation: {initial_score:.2f}, NNZ params: {initial_nnz}")
    
    # Initialize experience replay
    if use_per:
        exp_replay = Supervised_Prioritzed_ExperienceReplay(
            size=config.memory_size, alpha=config.ALPHA_PER
        )
    else:
        exp_replay = ExperienceReplay(size=config.memory_size)
    
    stop_pruning_flag = False
    low_score_consecutive = 0
    no_prune_consecutive = 0
    convergence_count = 0
    learning_rate_multiplier = 1.0
    plus = True
    multiplier = 10
    
    for iteration in range(iterations):
        logger.info(f"=== ITERATION {iteration+1}/{iterations}: Accumulating experience ===")
        print(f"=== ITERATION {iteration+1}/{iterations}: Accumulating experience ===")
        
        # Accumulate experience from teacher (use the student as teacher for self-improvement)
        accumulate_experience_sac_cartpole(
            teacher_agent=student_agent,
            exp_replay=exp_replay,
            num_steps=config.OBSERVE_PERIOD_EXPERIENCE,
            env=env
        )
        
        logger.info(f"=== ITERATION {iteration+1}/{iterations}: Training and pruning student ===")
        print(f"=== ITERATION {iteration+1}/{iterations}: Training and pruning student ===")
        
        # Dynamic learning rate
        current_lr_actor = (config.learning_rate_schedule_actor(iteration, config.lr_actor) 
                           * learning_rate_multiplier)
        current_lr_critic = (config.learning_rate_schedule_critic(iteration, config.lr_critic) 
                            * learning_rate_multiplier)
        
        # Train student with pruning
        score_list, nnz_params_list, stop_pruning_flag = train_sac_student_cartpole(
            logger=logger,
            student_agent=student_agent,
            exp_replay=exp_replay,
            env=env,
            prune_this_iteration=True,
            best_model_path=best_actor_path,
            lr_actor=current_lr_actor,
            lr_critic=current_lr_critic,
            stop_pruning_temporarily=stop_pruning_flag,
            use_per=use_per,
            current_pruning_iteration=iteration,
            num_train_steps=config.n_epoch * 200,  # Adjust as needed
            objective_score=objective_score,
            lower_bound_score=lower_bound_score,
            config=config
        )
        
        # Update tracking arrays
        for j, score in enumerate(score_list):
            if j < len(nnz_params_list) and nnz_params_list[j] < nnz_vs_accuracy[0][-1]:
                nnz_vs_accuracy[1].append(score)
                nnz_vs_accuracy[0].append(nnz_params_list[j])
        
        mean_score = np.mean(score_list) if score_list else 0
        logger.info(f"=== Iteration {iteration+1}: Mean score after pruning: {mean_score:.2f} ===")
        print(f"=== Iteration {iteration+1}: Mean score after pruning: {mean_score:.2f} ===")
        
        # Check stopping conditions
        if mean_score < lower_bound_score:
            low_score_consecutive += 1
            if low_score_consecutive >= 5:
                logger.info(f"Stopping due to low accuracy for 5 consecutive trials")
                break
        else:
            low_score_consecutive = 0
        
        if stop_pruning_flag:
            no_prune_consecutive += 1
            if no_prune_consecutive >= 5:
                logger.info("Adjusting learning rate due to pruning difficulties")
                if plus:
                    learning_rate_multiplier *= multiplier
                    plus = False
                    multiplier *= 10
                else:
                    learning_rate_multiplier /= multiplier
                    plus = True
                    multiplier *= 10
            
            # Check for convergence
            current_nnz = nnz_vs_accuracy[0][-1] if nnz_vs_accuracy[0] else initial_nnz
            if len(nnz_vs_accuracy[0]) > 1 and current_nnz == nnz_vs_accuracy[0][-2]:
                convergence_count += 1
                if convergence_count >= 5:
                    logger.info("Sparsity converged, ending pruning procedure")
                    break
            else:
                convergence_count = 0
        else:
            multiplier = 10.0
            no_prune_consecutive = 0
    
    return nnz_vs_accuracy


def check_convergence(info):
    """Check if the model size has converged"""
    if len(info) < 2:
        return False
    
    diff_temp = []
    for i in range(len(info) - 1):
        diff_temp.append(info[i] - info[i+1])
    
    mean_diff = sum(diff_temp) / len(diff_temp)
    return mean_diff < 0.05  # Less than 5% change


def copy_weights_teacher_to_student(teacher, student):
    """Copy weights from teacher model to student model using the framework's copy_weights method"""
    # Extract weights from teacher model (in teacher's session and graph context)
    with teacher.graph.as_default():
        teacher_vars = [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) 
                       if teacher.scope in v.name]
        teacher_weights = teacher.sess.run(teacher_vars)
    
    # Get student variables (excluding mask variables) for copying
    with student.graph.as_default():
        student_vars = [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) 
                       if student.scope in v.name and 'mask' not in v.name.lower()]
        
        # Ensure we have matching numbers of variables
        min_vars = min(len(teacher_weights), len(student_vars))
        
        copy_ops = []
        for i in range(min_vars):
            copy_ops.append(tf.assign(student_vars[i], teacher_weights[i]))
        
        if copy_ops:
            student.sess.run(copy_ops)


def main():
    """Main function for SAC pruning experiment"""
    # Setup logging
    logger = get_logger(FLAGS.PoPS_dir + "/SAC_Pruning_CartPole")
    logger.info("============= START SAC PRUNING =============")
    
    # Initialize environment
    env = gym.make('CartPole-v0')
    
    # Load teacher model (pre-trained SAC)
    logger.info("Loading pre-trained SAC teacher model...")
    teacher = CartPoleSAC(
        input_size=PruneCartpoleSACConfig.input_size,
        output_size=PruneCartpoleSACConfig.output_size,
        model_path=PruneCartpoleSACConfig.model_path_sac
    )
    teacher.load_model()
    teacher.init_target()
    
    # Evaluate teacher
    logger.info("Evaluating teacher model...")
    teacher_score = evaluate_sac_cartpole(teacher, n_epoch=FLAGS.eval_epochs)
    logger.info(f"Teacher evaluated with score: {teacher_score:.2f}")
    print(f"Teacher evaluated with score: {teacher_score:.2f}")
    
    # Create student model for pruning
    student_path = FLAGS.PoPS_dir + "/student_sac_initial"
    
    logger.info("Creating student model...")
    student = StudentCartPoleSAC(
        input_size=PruneCartpoleSACConfig.input_size,
        output_size=PruneCartpoleSACConfig.output_size,
        model_path=student_path,
        target_sparsity=PruneCartpoleSACConfig.target_sparsity,
        pruning_freq=PruneCartpoleSACConfig.pruning_freq,
        sparsity_end=PruneCartpoleSACConfig.sparsity_end
    )    # 權重複製 (使用框架的 copy_weights 方法)
    copy_weights_teacher_to_student(teacher, student)
    # --- 在 restore 或 copy 完 teacher 權重後，檢查 graph 中未初始化的變數 ---
    # 修正：確保 tf.report_uninitialized_variables() 在 student.graph 下產生
    with student.graph.as_default():
        uninit_tensors = student.sess.run(tf.report_uninitialized_variables())
   
    # 如果你看到類似 "StudentCartPoleSAC/q1/fc2/bias" 之類的變數還沒被初始化，
    # 代表除了 mask 之外，還有其他 network 權重／bias 也需要初始化。可以把他們一次初始化：
    if len(uninit_tensors) > 0:
        # 舉例：直接初始化所有未初始化的變數
        student.sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if v.name.split(':')[0] in 
            [n.decode("utf-8").split(":")[0] for n in uninit_tensors]]
        ))

    # 接著，再呼叫 init_target() 將 online 權重複製到 target
    student.init_target()

    initial_size = student.get_number_of_nnz_params()
    logger.info(f"Initial student model size: {initial_size} parameters")
    
    # Run iterative pruning
    logger.info("Starting iterative pruning procedure...")
    sparsity_vs_accuracy = iterative_pruning_sac_cartpole(
        logger=logger,
        student_agent=student,
        env=env,
        iterations=FLAGS.iterations,
        use_per=FLAGS.use_per,
        config=PruneCartpoleSACConfig,
        best_actor_path=FLAGS.best_path,
        objective_score=PruneCartpoleSACConfig.objective_score,
        lower_bound_score=PruneCartpoleSACConfig.lower_bound_score
    )
    
    # Plot results
    plot_graph(
        data=sparsity_vs_accuracy,
        name=FLAGS.PoPS_dir + "/SAC_pruning_results",
        figure_num=1,
        xaxis='NNZ Parameters',
        yaxis='Performance Score'
    )
    
    logger.info("============= SAC PRUNING COMPLETE =============")
    
    # Clean up
    env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--teacher_path',
        type=str,
        default=PruneCartpoleSACConfig.model_path_sac,
        help='Path to pre-trained SAC teacher model'
    )
    parser.add_argument(
        '--PoPS_dir',
        type=str,
        default=PruneCartpoleSACConfig.iterative_PoPS,
        help='Results directory'
    )
    parser.add_argument(
        '--best_path',
        type=str,
        default=PruneCartpoleSACConfig.prune_best_actor,
        help='Path to save best pruned model'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=PruneCartpoleSACConfig.n_pruning_iterations,
        help='Number of pruning iterations'
    )
    parser.add_argument(
        '--eval_epochs',
        type=int,
        default=100,
        help='Number of episodes for evaluation'
    )
    parser.add_argument(
        '--use_per',
        type=bool,
        default=False,
        help='Use Prioritized Experience Replay'
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    main()