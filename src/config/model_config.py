from src.config.ModelHyperparams import ModelHyperparams

model_config_dict = {
    'default': ModelHyperparams(),
    
    'test': ModelHyperparams(iters=100, stop_method='nll')
}
