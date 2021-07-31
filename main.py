from agent import Agent
from gym_wrapper import DonkeySimEnv

def trainFunction():   
    plot_scores = []
    training_evaluation_counter = 1
    train_ratio = 2
    record = 0
    agent = Agent()  # Initialize the Training Agent
    agent.games = 50 # Set Maximum Number of Games
    agent.Training = True
    game = DonkeySimEnv()  # Connect to the Simulator
    while agent.n_games < agent.games:  # Set Limit to how many games it will train for.
        if (training_evaluation_counter % train_ratio == 0):
            training_evaluation_counter = 0 # EVALUATION MODE
            print("SETTING DRIVING MODE: Evaluation")
            agent.training = False
        else:
            print("SETTING DRIVING MODE: Training")
            agent.training = True # TRAINING MODE
        state_old = agent.get_state(game)  # Get Old State
        final_move = agent.get_action(state_old)  # Get Move
        obs, reward, done, info, score = game.step(final_move)  # Perform action
        state_new = agent.get_state(game)  # Get new state
        agent.train_short_memory(state_old, final_move, reward, state_new, done)  # Train short memory
        agent.remember(state_old, final_move, reward, state_new, done)  # Remember
        if done:  # Train long term memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                    record = score
                    agent.model_save()

            # print('Training: Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            if (agent.training == False):
                print('Evaluation: Game', agent.n_games, 'Score', score, 'Record:', record)
            else:
                print('Training: Game', agent.n_games, 'Score', score, 'Record:', record)
            training_evaluation_counter += 1

if __name__ == '__main__':
    trainFunction()
