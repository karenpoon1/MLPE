from src.config.dataclasses.SplitParams import SplitParams

split_config_dict = {
    'default': SplitParams(random_q_order=True,
                           random_s_order=False,
                           test_split=0.2,
                           val_split=0.1),

    'test': SplitParams(random_q_order=False,
                        random_s_order=False,
                        test_split=0.2,
                        val_split=0.1),

    '85-10-5': SplitParams(random_s_order=False,                            
                           random_q_order=False,
                           test_split=0.1,
                           val_split=0.05)
}
