from tqdm import tqdm
from .blackjack import BlackjackEnv, Hand, Card

ACTIONS = ['hit', 'stick']

def policy_evaluation(env, V, policy, episodes=500000, gamma=1.0):
    """
    Monte Carlo policy evaluation:
    - Generate episodes using the current policy
    - Update state value function as an average return
    """
    # Track number of visits to each state and track sum of returns for each state
    return_sum = {}
    return_count = {}

    # Initialize returns_sum and returns_count

    for _ in tqdm(range(episodes), desc="Policy evaluation"):
        episode = []
        state = env.reset()
        done = False

        # Generate one episode
        while not done:
            action = policy[state]
            result = env.step(action)

            if len(result) == 3:
                next_state, reward, done = result
            elif len(result) == 4:
                next_state, reward, done, _ = result
            elif len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected result format from env.step: {result}")
                
            episode.append((state, action, reward))
            state = next_state

        # First-visit Monte Carlo: Update returns for the first occurrence of each state
            # Compute return from the first visit onward
                # Update returns_sum and returns_count
        G = 0
        visited_states = set()

        for i in range(len(episode)-1, -1, -1):
            state, action, reward = episode[i]
            G = gamma * G + reward
            
            if state not in visited_states:
                visited_states.add(state)
                if state not in return_sum:
                    return_sum[state] = 0
                    return_count[state] = 0
                return_sum[state] += G
                return_count[state] += 1

    # Update V(s) as the average return
    for state in return_sum:
        V[state] = return_sum[state] / return_count[state]

    return V
