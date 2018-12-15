competition_type = 'playoff'

description = """
Test different go engines
"""

record_games = True
stderr_to_log = False

players = {
    # gigo
    'gigo' : Player("./run_gigo.sh",
                    startup_gtp_commands=[
                        "time_settings 0 5 1"
                        ]),

    # leelaz
    'leelaz' : Player("./run_leelaz.sh",
                      startup_gtp_commands=[
                          "time_settings 0 7 1"
                          ]),
    # ELF
    'elf' : Player("./run_elf.sh",),

    # Fuego at 5000 playouts per move
    'fuego-5k' : Player("fuego --quiet",
                        startup_gtp_commands=[
                            "go_param timelimit 999999",
                            "uct_max_memory 350000000",
                            "uct_param_search number_threads 1",
                            "uct_param_player reuse_subtree 0",
                            "uct_param_player ponder 0",
                            "uct_param_player max_games 5000",
                            ]),
    }

board_size = 19
komi = 7.5

matchups = [
    Matchup('gigo',
            'leelaz',
            alternating=True,
            scorer='players', move_limit=723),

    ]
