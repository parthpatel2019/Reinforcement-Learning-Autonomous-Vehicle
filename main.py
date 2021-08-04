from agent import Agent
from gym_wrapper import DonkeySimEnv

def trainFunction():   
    plot_scores = []
    training_evaluation_counter = 2
    train_ratio = 5
    record = 0
    agent = Agent()  # Initialize the Training Agent
    agent.games = 50 # Set Maximum Number of Games
    agent.Training = True
    game = DonkeySimEnv()  # Connect to the Simulator
    print("SETTING DRIVING MODE: Training")
    while agent.n_games < agent.games:  # Set Limit to how many games it will train for.
        state_old = agent.get_state(game)  # Get Old State
        final_move = agent.get_action(state_old)  # Get Move
        obs, reward, done, score = game.step(final_move)  # Perform action
        if (agent.training == True):
            state_new = agent.get_state(game)  # Get new state
            agent.train_short_memory(state_old, final_move, reward, state_new, done)  # Train short memory
            agent.remember(state_old, final_move, reward, state_new, done)  # Remember
        if done:  # Train long term memory
            game.reset()
            agent.n_games += 1
            if (agent.training == True):
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
            if (training_evaluation_counter % train_ratio == 0):
                training_evaluation_counter = 0  # EVALUATION MODE
                print("SETTING DRIVING MODE: Evaluation")
                agent.training = False
            elif(training_evaluation_counter == 1 and len(plot_scores) != 1):
                print("SETTING DRIVING MODE: Training")
                agent.training = True  # TRAINING MODE
            game.teleport()

if __name__ == '__main__':
    trainFunction()
