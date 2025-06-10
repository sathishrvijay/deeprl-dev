import gymnasium as gym

VERBOSE=0

if __name__ == "__main__":
    # instantiate the existing Cartpole environment. Randomly sample action at each time step
    # Environment returns a reward of 1 for every time step that cartpole remains not fallen
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Turns on rendering of CartPole for visualization
    env = gym.wrappers.HumanRendering(env)

    episode_return = 0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()

        obs, reward, done, _, _ = env.step(action)
        if VERBOSE:
            print(f"step {total_steps}; action taken: {action} & reward: {reward:.2f}")
            print(f"observation: {obs}")

        total_steps += 1
        episode_return += reward
        if done:
            # episode has ended because CartPole has fallen
            break


    print(f"Episode completed in {total_steps}; Total reward: {episode_return:.2f}")

    # Required only if rendering i
    env.close()
