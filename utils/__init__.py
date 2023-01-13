from utils.simclr import simclr_loss_func
from utils.barlow_twin import barlow_loss_func

SELF_SUPERVISED_LOSS_FUNC = {'simclr':simclr_loss_func,
                             'barlow_twins':barlow_loss_func}
