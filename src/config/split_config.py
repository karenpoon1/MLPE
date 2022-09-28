from src.config.SplitParams import SplitParams

split_config_dict = {
    'default': SplitParams(),

    'test': SplitParams(random_q_order=False)
}
