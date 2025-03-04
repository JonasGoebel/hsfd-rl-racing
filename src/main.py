from RacingGame import RacingGame

from models.Action import Action


def main():
    game_size_pixels = (1000, 1000)
    game = RacingGame(game_size_pixels)

    is_done = False
    while not is_done:
        action = game.handle_events()

        new_state, reward, is_done = game.step(action)
        game.render()


if __name__ == "__main__":
    main()
