from configs.loftr.eloftr_full import cfg

cfg.LOFTR.LOSS.LOCAL_LOSS_TYPE = 'nll_gaussian'   # or 'nll_laplace'
cfg.LOFTR.MATCH_FINE.PREDICT_VAR = True            # learned head; False = nothing to train, skip to Step 3
