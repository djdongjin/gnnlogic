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
    'data_1.2,1.3': '6d433116803640ffbe67375aa03df3ac',
    'data_1.3,1.4': '83589fa92bef4dd28432d9f9cde37b06',
    'data_1.4,1.5': '2ac8c6c9935b4397bd786eeec3144fc5',
    'data_1.5,1.6': 'f7a1de2dee65468cbc7824994d2f039c',
    'data_1.6,1.7': 'e1d21726e95d4bb6b35f8a48e03eff2d',
    'data_1.7,1.8': 'edef5789d67e436d988fc5bfe9da1170',
    'data_1.8,1.9': '',
    'data_1.9,1.10': '',
    'data_1.2,1.3_clean': 'eea017269cfb4606b4e50603b3af50e1',
    'data_4.2,4.3_disconnected': '6570da6fe65b45668b826827df2cd4e2',
    'data_3.2,3.3_irrelevant':'043f5959d60d4fd589f2120b5cd26a6a',
    'data_2.2,2.3_supporting': 'bc00650921dd461aa990fa8e7965bf37',

}

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
    config_id, exp_id, dataset = argument_parser()
    print(config_id)
    if len(exp_id) == 0:
        config = get_config(config_id=config_id)
        if dataset:
            config['dataset']['data_path'] = dataset
        # TODO: hot fix to automate pretrained model loading
        if 'kd' in config_id or 'max' in config_id:
            config['model']['dual']['teacher_exp_id'] = pretrained_task_to_exp_id[dataset]
        log_base = config['general']['base_path']
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_base, 'logs', "{0}.log".format(config_id))),
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
        name = 'exp_{}'.format(config_id)
        config.general.exp_name = name
        ex.log_parameters(flatten_dictionary(config))
        ex.set_name(name)
        start(config, ex)
    else:
        logging.info("Resuming old experiment with id {}".format(exp_id))
        config = get_config(config_id=config_id)
        if dataset:
            config['dataset']['data_path'] = dataset
        logger = logging.getLogger()
        ex = ExistingExperiment(
            api_key=config.log.comet.api_key,
            previous_experiment=exp_id,
            workspace=config.log.comet.workspace,
            project_name=config.log.comet.project_name,
            disabled=config.log.comet.disabled,
            auto_output_logging=None,
            log_code=False,)
        name = 'exp_{}'.format(config_id)
        config.general.exp_name = name
        resume(config, ex)
