import json
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class ChemVAETrainingParams:
    """Initialise and store the parameters for training the ChemVAE model."""

    # data parameters
    data_size_for_all_loops: int = None   # number of molecules in the dataset
    data_width: int = 120  # previous name: MAX_LEN
    data_height: int = 35  # previous name: NCHARS
    loop_over_fit_batch_size: int = 1260
    loop_over_fit_batch_id: int = None
    paired_output: bool = False  # whether to use paired output for the decoder
    RAND_SEED: int = None
    PADDING: str = "right"

    # for starting model from a checkpoint
    reload_model: bool = False
    prev_epochs: int = 0

    # general parameters
    model_fit_batch_size: int = 100
    epochs: int = 1
    val_split: float = 0.1  # validation split fraction
    loss: str = "categorical_crossentropy"  # set reconstruction loss
    if_stratify: bool = False  # whether to stratify the data for training
    if_smiles: bool = True  # whether to use SMILES strings for training

    # convolution parameters
    batchnorm_conv: bool = True
    conv_activation: str = "relu"
    conv_depth: int = 4
    conv_dim_depth: int = 8
    conv_dim_width: int = 8
    conv_d_growth_factor: float = 1.15875438383
    conv_w_growth_factor: float = 1.1758149644

    # decoder parameters
    gru_depth: int = 4
    rnn_activation: str = "tanh"
    recurrent_dim: int = 488
    do_tgru: bool = False
    terminal_GRU_implementation: int = 0
    tgru_dropout: float = 0.0
    temperature: float = 1.00

    # middle layer parameters
    hg_growth_factor: float = 1.4928245388
    hidden_dim: int = 196
    middle_layer: int = 1
    dropout_rate_mid: float = 0.0
    batchnorm_mid: bool = True  # apply batch normalization to middle layers
    activation: str = "relu"

    # Optimization parameters
    lr: float = 0.000312087049936
    momentum: float = 0.936948773087
    optim: str = "adam"

    # vae parameters
    vae_annealer_start: int = 22  # Center for variational weigh annealer
    batchnorm_vae: bool = False  # apply batch normalization to output of the variational layer
    vae_activation: str = "relu"
    xent_loss_weight: float = 1.0  # loss weight to assign to reconstruction error.
    kl_loss_weight: float = 1.0  # loss weight to assign to KL loss
    anneal_sigmod_slope: float = 1.0  # slope of sigmoid variational weight annealer
    freeze_logvar_layer: bool = False  # Choice of freezing the variational layer until close to the anneal starting epoch
    freeze_offset: int = 1  # the number of epochs before vae_annealer_start where the variational layer should be unfrozen

    # property prediction parameters
    # do_prop_pred: bool = False  # whether to do property prediction
    # prop_pred_depth: int = 3
    # prop_hidden_dim: int = 36
    # prop_growth_factor: float = 0.8  # slope of sigmoid variational weight annealer
    # prop_pred_activation: str = "relu"
    # reg_prop_pred_loss: str = "mse"  # loss function to use with property prediction error for regression tasks
    # logit_prop_pred_loss: str = "binary_crossentropy"  # loss function to use with property prediction for logistic tasks
    # prop_pred_loss_weight: float = 0.5

    # print output parameters
    verbose_print: int = 0

    # gpu parameters
    n_gpu: int = 1

    # files
    directory: str = None  # parent directory for all files
    data_file: str = None  # data file name/path
    vae_weights_file: str = None  # model weights file name/path
    test_idx_file: str = None
    history_file: str = None
    checkpoint_path: str = None
    limit_data: int = None
    char_file: str = None


def load_params(param_file=None, verbose=True):
    # Parameters from params.json and exp.json and update the dataclass ChemVAETrainingParams with them.
    # default parameters
    training_params = ChemVAETrainingParams()

    if param_file is not None:
        hyper_p = json.loads(open(param_file).read(),
                             object_pairs_hook=OrderedDict)
        if verbose:
            print("Using hyper-parameters:")
            for key, value in hyper_p.items():
                print("{:25s} - {:12}".format(key, str(value)))
            print("rest of parameters are set as default")

        # overwrite the default parameters in the dataclass
        training_params.__dict__.update(hyper_p)

    return training_params

    # temporary renaming
    # parameters["model_fit_batch_size"] = parameters["batch_size"]
    # parameters["loop_over_fit_batch_size"] = parameters["training_batch_size"]  # subset data for training per training run
    # parameters["loop_over_fit_batch_id"] = parameters["batch_id"]
    # parameters["data_size_for_all_loops"] = parameters["data_size"]
