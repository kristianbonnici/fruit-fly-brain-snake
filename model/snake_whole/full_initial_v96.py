"""The same verified V90 efficacy, with a new objective and fresh optimizer."""
from .full_initial_v95 import initial_efficacy as original,verify_initial_predictions


def initial_efficacy():
    efficacy,source=original()
    source['new_optimizer']='Fresh zero Adam moments for rare-fatal teacher CE and source-retention KL'
    source['training_run']='Objective variant from preserved V90run91001; not an independent training replicate'
    return efficacy,source
