import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'predictor':
        from .P_model import P_Model as M
    elif model == 'corrector':
        from .C_model import C_Model as M
    elif model == 'sftmd':
        from .F_model import F_Model as M
    elif model == 'pred_EncDec':
        from .PEncDec_model import PEncDec_Model as M
    elif model == 'corr_EncDec':
        from .CEncDec_model import CEncDec_Model as M
    elif (model == 'ker_EncDec') | (model == 'ker_Enc') | (model == 'ker_Dec') | (model == 'DAN_ker'):
        from .CEncDec_model import CEncDec_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
