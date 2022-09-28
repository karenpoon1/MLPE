from src.config.ModelParams import ModelParams

model_config_dict = {
    'default': ModelParams(),
    
    'test': ModelParams(iters=100, stop_method='nll')
}
