import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ==========================================
# 0. UTILITAIRES
# ==========================================

def normalize_angle(angle):
    """Normalise l'angle entre -pi et pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ==========================================
# 1. ENVIRONNEMENT OTT (Optimized Target Tracking)
# ==========================================

from gym_gmmcar.envs.circle_env import CircleEnv

class OttEnv(CircleEnv):
    """
    Environnement optimisé pour la fluidité (Smoothness) et la vitesse.
    Reward V15 "Anti-Yoyo" intégrée.
    """
    def __init__(self, target_velocity=1.0, radius=1.0, dt=0.035, model_type='BrushTireModel', 
                 robot_type='RCCar', mu_s=1.37, mu_k=1.96, eps=0.05):
        super().__init__(target_velocity=target_velocity, radius=radius, dt=dt, 
                         model_type=model_type, robot_type=robot_type, mu_s=mu_s, mu_k=mu_k)
        
        self.eps = eps
        self.prev_action = None
        self.next_waypoint_idx = 0
        
        # Waypoints (pour visualisation uniquement)
        r = self.radius
        self.waypoints = [(0, -r), (r, 0), (0, r), (-r, 0)]
        
        # Observation Space (6 dimensions)
        import gym
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.prev_action = None
        self.next_waypoint_idx = 0
        state = super().reset()
        return self.state_to_observation(state)

    def get_reward(self, state, action):
        x, y, yaw, x_dot, y_dot, yaw_dot = state
        
        # --- 1. DONNÉES PHYSIQUES ---
        dist_center = np.sqrt(x**2 + y**2)
        dist_error = np.abs(dist_center - self.radius)
        
        # Vitesse Tangentielle (Pour éviter qu'il apprenne à reculer ou tourner sur place)
        radius_safe = dist_center if dist_center > 0.05 else 0.05
        tangential_speed = (x * y_dot - y * x_dot) / radius_safe
        
        # --- 2. SÉCURITÉ (MUR DE LAVE) ---
        if dist_center < 0.6 or dist_center > 2.0:
            return -10.0, {} 
        
        # Clapet Anti-Retour (Sens horaire interdit)
        if tangential_speed < -0.1:
            return -2.0 * np.abs(tangential_speed), {}

        # --- 3. REWARD "ADDITIVE" (Fluidité) ---
        
        # A. POSITION (Indépendant)
        # On garde votre formule exponentielle, elle est très bien.
        reward_position = np.exp(-2.0 * dist_error)
        
        # B. ALIGNEMENT (Indépendant)
        angle_pos = np.arctan2(y, x)
        track_dir = angle_pos + (np.pi / 2)
        heading_err = (track_dir - yaw + np.pi) % (2 * np.pi) - np.pi
        alignment = np.cos(heading_err)
        reward_alignment = max(0.0, alignment)
        
        # C. VITESSE (Indépendant et Continu !)
        # PLUS DE CONDITION "is_on_track".
        # La vitesse paie toujours, mais la position paie aussi.
        target_v = self.target_velocity if hasattr(self, 'target_velocity') else 1.0
        
        if tangential_speed > 0:
            # On additionne (+) au lieu de multiplier (*).
            # Cela évite que le reward s'effondre si un seul paramètre faiblit.
            reward_speed = 1.5 * (tangential_speed / target_v)
        else:
            reward_speed = 0.0

        # --- 4. STABILITÉ (Le Secret du Lissage) ---
        steering = action[1]
        
        # Anti-Yoyo (Derivée du volant)
        # On punit le CHANGEMENT d'angle. C'est ça qui lisse la courbe.
        delta_steer = 0.0
        if self.prev_action is not None:
            delta_steer = np.abs(steering - self.prev_action[1])
        
        reward_smoothness = -2.0 * delta_steer

        # --- TOTAL (Somme Pondérée) ---
        # Position + Vitesse + Alignement - Secousses
        reward = (2.0 * reward_position) + (1.0 * reward_speed) + (1.0 * reward_alignment)
        
        # On soustrait les pénalités
        reward += reward_smoothness + 0.1

        self.prev_action = action
        return reward, {}

    def state_to_observation(self, state):
        r = self.radius
        x, y, yaw, x_dot, y_dot, yaw_dot = state
        
        dist_center = np.sqrt(x**2 + y**2)
        dx = dist_center - r
        
        angle_pos = np.arctan2(y, x)
        desired_yaw = angle_pos + (np.pi / 2)
        theta = normalize_angle(desired_yaw - yaw)

        ddx = (x * x_dot + y * y_dot) / (dist_center + 1e-6)
        dtheta = -yaw_dot
        cos_angle = x / (dist_center + 1e-6)
        sin_angle = y / (dist_center + 1e-6)

        return np.array([dx, theta, ddx, dtheta, cos_angle, sin_angle], dtype=np.float32)

    def step(self, action):
        next_state, reward, super_done, info = super().step(action)
        observation = self.state_to_observation(next_state)
        
        extra_done = self._get_done(next_state)
        done = super_done or extra_done
        
        my_reward, my_info = self.get_reward(next_state, action)
        
        if info is None: info = {}
        info.update(my_info)
        
        if extra_done:
            my_reward -= 5.0
            
        return observation, my_reward, done, info

    def _get_done(self, state):
        # Synchronisé avec le reward de "Lave"
        x, y = state[0], state[1]
        dist_center = np.sqrt(x**2 + y**2)
        
        too_close = dist_center < 0.6
        too_far = dist_center > 2.0
        
        return too_close or too_far

# ==========================================
# 2. ALGORITHME DDPG (Actor-Critic)
# ==========================================

class Actor:
    def __init__(self, state_dim, action_dim, action_bound_v, action_bound_alpha, neurons):
        self.model = self._build_model(state_dim, action_dim, neurons, action_bound_v, action_bound_alpha)

    def _build_model(self, state_dim, action_dim, neurons, bound_v, bound_alpha):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(neurons, activation="relu")(inputs)
        x = layers.Dense(neurons, activation="relu")(x)
        
        # Vitesse (Sigmoid -> [0, V_MAX])
        raw_v = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init, name="out_velocity")(x)
        out_v = layers.Lambda(lambda z: z * bound_v)(raw_v)
        
        # Volant (Tanh -> [-Angle, +Angle])
        raw_alpha = layers.Dense(1, activation="tanh", kernel_initializer=last_init, name="out_steering")(x)
        out_alpha = layers.Lambda(lambda z: z * bound_alpha)(raw_alpha)
        
        outputs = layers.Concatenate()([out_v, out_alpha])
        return tf.keras.Model(inputs, outputs)

class Critic:
    def __init__(self, state_dim, action_dim, neurons):
        self.model = self._build_model(state_dim, action_dim, neurons)

    def _build_model(self, state_dim, action_dim, neurons):
        state_input = layers.Input(shape=(state_dim,))
        action_input = layers.Input(shape=(action_dim,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)
        action_out = layers.Dense(32, activation="relu")(action_input)
        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(neurons, activation="relu")(concat)
        out = layers.Dense(neurons, activation="relu")(out)
        outputs = layers.Dense(1)(out)
        return tf.keras.Model([state_input, action_input], outputs)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, state_dim=6, action_dim=2):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch,
               actor_model, critic_model, target_actor, target_critic, 
               gamma, actor_optimizer, critic_optimizer):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True) * (1 - done_batch)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    def learn(self, actor_model, critic_model, target_actor, target_critic, 
              gamma, actor_optimizer, critic_optimizer):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices], dtype=tf.float32)
        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                    actor_model, critic_model, target_actor, target_critic, 
                    gamma, actor_optimizer, critic_optimizer)

def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound_v, action_bound_alpha, 
                 lra=0.0003, lrc=0.001, gamma=0.99, tau=0.005, sigma=0.2, neurons=256):
        self.actor = Actor(state_dim, action_dim, action_bound_v, action_bound_alpha, neurons)
        self.critic = Critic(state_dim, action_dim, neurons)
        self.target_actor = Actor(state_dim, action_dim, action_bound_v, action_bound_alpha, neurons)
        self.target_critic = Critic(state_dim, action_dim, neurons)
        self.target_actor.model.set_weights(self.actor.model.get_weights())
        self.target_critic.model.set_weights(self.critic.model.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(lra)
        self.critic_optimizer = tf.keras.optimizers.Adam(lrc)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma

    def policy(self, state, noise_object=None):
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            sampled_actions = tf.squeeze(self.actor.model(tf_state))
            
            if noise_object is not None:
                noise = noise_object()
                sampled_actions = sampled_actions.numpy() + noise
            else:
                sampled_actions = sampled_actions.numpy()
            
            lower_bounds = [0.0, -np.pi/6]
            upper_bounds = [10.0, np.pi/6]
            
            return np.clip(sampled_actions, lower_bounds, upper_bounds)

class OUActionNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + 
             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x
    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

# ==========================================
# 3. BOUCLE D'ENTRAINEMENT
# ==========================================

def train(episodes=100, buffer_size=100000, batch_size=128, gamma=0.99, tau=0.005, 
          lra=0.00005,  # <--- CLÉ DU SUCCÈS : Learning Rate très bas pour le lissage
          lrc=0.001, sigma=0.1, neurons=256, 
          min_buffer=5000, # <--- CLÉ DU SUCCÈS : Gros buffer initial
          load_weights=True, save_prefix="../weights/"):
    
    # On force une vitesse cible élevée pour permettre à l'agent d'accélérer
    env = OttEnv(radius=1.0, target_velocity=5.0)
    
    # --- BRUIT ASYMÉTRIQUE (LE SILENCE DU VOLANT) ---
    # Vitesse : 1.0 (Exploration nécessaire)
    # Volant : 0.05 (Très calme pour éviter le yoyo)
    std_dev = np.array([1.0, 0.05])
    ou_noise = OUActionNoise(mean=np.zeros(2), std_dev=std_dev)

    agent = DDPGAgent(6, 2, 10.0, np.pi/6, lra, lrc, gamma, tau, sigma, neurons)
    buffer = Buffer(buffer_size, batch_size, 6, 2)

    if load_weights:
        try:
            agent.actor.model.load_weights(f"{save_prefix}actor_model.weights.h5")
            agent.critic.model.load_weights(f"{save_prefix}critic_model.weights.h5")
        except: print("Poids non trouvés, démarrage à zéro.")

    reward_history, avg_reward_history = [], []

    for ep in range(episodes):
        prev_state = env.reset()
        episodic_reward = 0
        step = 0

        while True:
            action = agent.policy(prev_state, ou_noise)
            state, reward, done, info = env.step(action)
            buffer.record((prev_state, action, reward, state, done))
            episodic_reward += reward

            if buffer.buffer_counter > min_buffer:
                buffer.learn(agent.actor.model, agent.critic.model, 
                             agent.target_actor.model, agent.target_critic.model, 
                             gamma, agent.actor_optimizer, agent.critic_optimizer)
                update_target(agent.target_actor.model.trainable_variables, 
                              agent.actor.model.trainable_variables, tau)
                update_target(agent.target_critic.model.trainable_variables, 
                              agent.critic.model.trainable_variables, tau)

            if done: break
            prev_state = state
            step += 1
            if step > 500: break

        reward_history.append(episodic_reward)
        avg_reward = np.mean(reward_history[-20:])
        avg_reward_history.append(avg_reward)
        print(f"Episode {ep+1} | Reward: {episodic_reward:.2f} | Avg: {avg_reward:.2f} | Steps: {step}")

        if (ep + 1) % 50 == 0:
            agent.actor.model.save_weights(f"{save_prefix}actor_model.weights.h5")
            agent.critic.model.save_weights(f"{save_prefix}critic_model.weights.h5")

    print("Saving final weights...")
    agent.actor.model.save_weights(f"{save_prefix}actor_model.weights.h5")
    agent.critic.model.save_weights(f"{save_prefix}critic_model.weights.h5")
    
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Reward", alpha=0.3)
    plt.plot(avg_reward_history, label="Moving Average (20)")
    plt.title(f"Training Progress ({save_prefix})")
    plt.savefig(f"{save_prefix}training_results.png")
    return reward_history, avg_reward_history, agent

if __name__ == "__main__":
    # Lance l'entrainement par défaut si exécuté directement
    train(episodes=100)