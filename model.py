from cscnet import ConvLista_T,  ListaParams

def build_model(cfg):
    params = ListaParams(cfg['model_cfg']['kernel_size'], cfg['model_cfg']['num_filters'], cfg['model_cfg']['stride'], 
        cfg['model_cfg']['num_iter'], cfg['model_cfg']['channels'])
    net = ConvLista_T(params, threshold=cfg['model_cfg']['threshold'], norm=cfg['model_cfg']['norm'])
    return net
    