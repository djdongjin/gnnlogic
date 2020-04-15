import random

from comet_ml import Experiment, ExistingExperiment
from codes.experiment.experiment import run_experiment
from codes.utils.config import get_config
from codes.utils.util import set_seed, flatten_dictionary
from codes.utils.argument_parser import argument_parser
from addict import Dict
import os
import logging
import sys
import signal
import socket

base_path = os.path.dirname(os.path.realpath(__file__)).split('codes')[0]

pretrained_task_to_exp_id = {
    'data_1.2,1.3': 'babedcb6ffc04b6cb2789844bc618668',
    'data_1.3,1.4': 'bb9f7fd7377447de84b24211e2da2d76',
    'data_1.4,1.5': '7e17c1bca74649b0bf76f971bc22f7cd',
    'data_1.5,1.6': 'c633a893971b49a3864ba59a163933c4',
    'data_1.6,1.7': '9aee757e846e474c891689ba0450b3a3',
    'data_1.7,1.8': '495cb3b5d010416d9fe2b43a87f5c4ed',
    'data_1.8,1.9': 'ff9f88262b49409a909a871c8505b54e',
    'data_1.9,1.10': '580933c4320144ee88939c696496852a',
    'data_1.2,1.3_clean': 'a17010702e0649a4b57b37b0236b821f',
    'data_2.2,2.3_supporting': 'c05f707f8c2944059e744854026a8e8e',
    'data_3.2,3.3_irrelevant':'6a749b3154284f28b854c9d0526d4ebb',
    'data_4.2,4.3_disconnected': 'c5ac090da80f42debc7bcf5f9bbbec82',
}

seed_candidate = [111, 222, 333, 444, 555, 666, 777, 888, 999, 0]

def start(config, experiment):
    config = Dict(config)
    set_seed(seed=config.general.seed)
    run_experiment(config, experiment)

def resume(config, experiment):
    config = Dict(config)
    set_seed(seed=config.general.seed)
    run_experiment(config, experiment, resume=True)

# SLURM REQUEUE LOGIC
def get_job_id():
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        return '%s_%s' % (os.environ['SLURM_ARRAY_JOB_ID'],
                          os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        return os.environ['SLURM_JOB_ID']



def requeue_myself():
    job_id = get_job_id()
    logging.warning("Requeuing job %s", job_id)
    os.system('scontrol requeue %s' % job_id)



def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    job_id = get_job_id()
    logging.warning(
        "Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        requeue_myself()
    else:
        logging.warning("Not the master process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logging.warning("Signal handler called with signal " + str(signum))
    logging.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    logging.warning("Signal handler installed.")

if __name__ == '__main__':
    init_signal_handler()
    args = argument_parser()
    print(args.config_id)
    if len(args.exp_id) == 0:
        config = get_config(config_id=args.config_id)
        if args.dataset:
            config['dataset']['data_path'] = args.dataset
            config['general']['seed'] = seed_candidate[args.seed_index]
        # TODO: hot fix to automate pretrained model loading
        if 'kd' in args.config_id or 'infomax' in args.config_id:
            config['model']['dual']['teacher_exp_id'] = pretrained_task_to_exp_id[config['dataset']['data_path']]
        log_base = config['general']['base_path']
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_base, 'logs', "{0}.log".format(args.config_id))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        logger.info("Running new experiment")
        ex = Experiment(api_key=config.log.comet.api_key,
                        workspace=config.log.comet.workspace,
                        project_name=config.log.comet.project_name,
                        disabled=config.log.comet.disabled,
                        # auto_output_logging=None,
                        log_code=False)
        name = 'exp_{}'.format(args.config_id)
        config.general.exp_name = name
        ex.log_parameters(flatten_dictionary(config))
        ex.set_name(name)
        start(config, ex)
    else:
        logging.info("Resuming old experiment with id {}".format(args.exp_id))
        config = get_config(config_id=args.config_id)
        if args.dataset:
            config['dataset']['data_path'] = args.dataset
        logger = logging.getLogger()
        ex = ExistingExperiment(
            api_key=config.log.comet.api_key,
            previous_experiment=args.exp_id,
            workspace=config.log.comet.workspace,
            project_name=config.log.comet.project_name,
            disabled=config.log.comet.disabled,
            auto_output_logging=None,
            log_code=False,)
        name = 'exp_{}'.format(args.config_id)
        config.general.exp_name = name
        resume(config, ex)
