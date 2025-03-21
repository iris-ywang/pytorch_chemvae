import torch
import torch.nn as nn
from chemvae_train.load_params import ChemVAETrainingParams
from chemvae_train.models_utils import add_activation, ManualGRUWithSoftmax


class FPEncoder(nn.Module):
    def __init__(self, params: ChemVAETrainingParams):
        super(FPEncoder, self).__init__()
        self.params = params
        self.siamese_layers = nn.ModuleList()
        self.middle_layers = nn.ModuleList()

        siamse_dim_list = []
        siamse_dim_list.append(params.data_width)

        self.siamese_layers.append(nn.Linear(
            in_features=params.data_width,
            out_features=int(params.data_width * params.fp_hidden_dim_reduction_rate),
        ))
        siamese_previous_n_out_features = int(params.data_width * params.fp_hidden_dim_reduction_rate)
        siamse_dim_list.append(siamese_previous_n_out_features)

        # add activation function
        self.siamese_layers.append(add_activation(params.fp_activation))
        if params.fp_dropout_rate > 0.0:
            self.siamese_layers.append(nn.Dropout(params.fp_dropout_rate))
        if params.batchnorm_conv:
            self.siamese_layers.append(nn.BatchNorm1d(
                num_features=siamese_previous_n_out_features))

        for i in range(params.fp_siamese_depth - 1):
            self.siamese_layers.append(nn.Linear(
                in_features=siamese_previous_n_out_features,
                out_features=int(siamese_previous_n_out_features * params.fp_hidden_dim_reduction_rate),
            ))

            siamese_previous_n_out_features = int(siamese_previous_n_out_features * params.fp_hidden_dim_reduction_rate)
            siamse_dim_list.append(siamese_previous_n_out_features) #[1024, 512, 256, 128]

            # add activation function
            self.siamese_layers.append(add_activation(params.fp_activation))
            if params.fp_dropout_rate > 0.0:
                self.siamese_layers.append(nn.Dropout(params.fp_dropout_rate))
            if params.batchnorm_conv:
                self.siamese_layers.append(nn.BatchNorm1d(
                    num_features=siamese_previous_n_out_features))


        # Middle layers
        concat_features = siamese_previous_n_out_features * 3
        middle_dim_list = [concat_features]
        for i in range(params.fp_concat_depth - 1):
            self.middle_layers.append(nn.Linear(
                in_features=concat_features,
                out_features=int(concat_features * params.fp_hidden_dim_reduction_rate),
            ))
            concat_features = int(concat_features * params.fp_hidden_dim_reduction_rate)
            middle_dim_list.append(concat_features)

            # add activation function
            self.middle_layers.append(add_activation(params.fp_activation))

            if params.dropout_rate_mid > 0:
                self.middle_layers.append(nn.Dropout(params.fp_dropout_rate))
            if params.batchnorm_mid:
                self.middle_layers.append(nn.BatchNorm1d(
                    num_features=concat_features))

        # Final layers: z_mean
        self.z_mean = nn.Linear(concat_features, params.hidden_dim)
        self.z_log_var = nn.Linear(concat_features, params.hidden_dim)

        middle_dim_list.append(params.hidden_dim)

        if self.params.batchnorm_vae:
            self.z_samp = nn.BatchNorm1d(num_features=self.params.hidden_dim)

        params.fp_siamese_dim_list = siamse_dim_list
        params.fp_middle_dim_list = middle_dim_list

        dummy_input = torch.zeros(5, params.data_width, 2)
        output_size = self._get_flattened_size(dummy_input)
        print(output_size)

    def _get_flattened_size(self, x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]

        # Pass the tensor through the Siamese layers first
        for siamese in self.siamese_layers:
            x1 = siamese(x1)
            x2 = siamese(x2)
            print(x1.size())

        # Concatenate the two tensors and their difference into a single tensor
        x = torch.cat((x1, x2, x1 - x2), 1)
        print(x.size())
        # Pass the concatenated tensor through the normal layers
        for norm in self.middle_layers:
            x = norm(x)
            print(x1.size())
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z_samp = variational_layers(z_mean, z_log_var)
        if self.params.batchnorm_vae:
            z_samp = self.z_samp(z_samp)

        # concatenate z_mean and z_log_var
        z_mean_log_var_output = torch.cat((z_mean, z_log_var), 1)

        return x.size()

    def forward(self, x):
        # get x1, x2 from x by the last dimension which should only be 2.
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]

        # Pass the tensor through the Siamese layers first
        for siamese in self.siamese_layers:
            x1 = siamese(x1)
            x2 = siamese(x2)

        # Concatenate the two tensors and their difference into a single tensor
        x = torch.cat((x1, x2, x1 - x2), 1)

        # Pass the concatenated tensor through the normal layers
        for norm in self.middle_layers:
            x = norm(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z_samp = variational_layers(z_mean, z_log_var)
        if self.params.batchnorm_vae:
            z_samp = self.z_samp(z_samp)

        # concatenate z_mean and z_log_var
        z_mean_log_var_output = torch.cat((z_mean, z_log_var), 1)
        return z_samp, z_mean_log_var_output


class FPDecoder(nn.Module):
    def __init__(self, params):
        super(FPDecoder, self).__init__()
        self.params = params
        self.middle_layers = nn.ModuleList()
        self.siamese_layers = nn.ModuleList()

        middle_dim_list = params.fp_middle_dim_list
        siamese_dim_list = params.fp_siamese_dim_list

        # reverse the FPEncoder's layers in the reverse order
        for i in range(params.fp_concat_depth):
            self.middle_layers.append(nn.Linear(
                in_features=middle_dim_list[-(i+1)],
                out_features=middle_dim_list[-(i+2)],
            ))
            self.middle_layers.append(add_activation(params.fp_activation))
            if params.dropout_rate_mid > 0:
                self.middle_layers.append(nn.Dropout(params.fp_dropout_rate))
            if params.batchnorm_mid:
                self.middle_layers.append(nn.BatchNorm1d(
                    num_features=middle_dim_list[-(i+2)]))

        # Siamese layers
        for i in range(params.fp_siamese_depth):
            self.siamese_layers.append(nn.Linear(
                in_features=siamese_dim_list[-(i+1)],
                out_features=siamese_dim_list[-(i+2)],
            ))

            # add activation function
            self.siamese_layers.append(add_activation(params.fp_activation))
            if params.fp_dropout_rate > 0.0:
                self.siamese_layers.append(nn.Dropout(params.fp_dropout_rate))
            if params.batchnorm_conv:
                self.siamese_layers.append(nn.BatchNorm1d(
                    num_features=siamese_dim_list[-(i+2)]))

        dummy_input = torch.zeros(5, params.hidden_dim)
        output_size = self._get_final_size(dummy_input)
        print(output_size)

    def forward(self, z):
        # Pass latent rep through middle layers first
        for norm in self.middle_layers:
            z = norm(z)

        # Slice the tensor into the three parts, x1, x2, and x1 - x2
        intermediate_size = self.params.fp_siamese_dim_list[-1]
        x1 = z[:, :intermediate_size]
        x2 = z[:, intermediate_size:2 * intermediate_size]
        x_diff = z[:, 2 * intermediate_size:]

        # Pass x1, x2 through the Siamese layers first
        for siamese in self.siamese_layers:
            x1 = siamese(x1)
            x2 = siamese(x2)

        # Stack x1 and x2 of shape (n_samples, params.data_length) in a third dimension
        x = torch.stack((x1, x2), 2)

        return x

    def _get_final_size(self, z):
        # Pass latent rep through middle layers first
        for norm in self.middle_layers:
            z = norm(z)

        # Slice the tensor into the three parts, x1, x2, and x1 - x2
        intermediate_size = self.params.fp_siamese_dim_list[-1]
        x1 = z[:, :intermediate_size]
        x2 = z[:, intermediate_size:2 * intermediate_size]
        x_diff = z[:, 2 * intermediate_size:]

        # Pass x1, x2 through the Siamese layers first
        for siamese in self.siamese_layers:
            x1 = siamese(x1)
            x2 = siamese(x2)

        # Stack x1 and x2 of shape (n_samples, params.data_length) in a third dimension
        x = torch.stack((x1, x2), 2)

        return x.size()

def variational_layers(z_mean, z_log_var):
    epsilon = torch.randn(z_mean.size(), device=z_mean.device)
    z_rand = z_mean + torch.exp(z_log_var / 2) * epsilon
    return z_rand


class FPVAEAutoEncoder(nn.Module):
    def __init__(self, params):
        super(FPVAEAutoEncoder, self).__init__()
        self.encoder = FPEncoder(params)
        self.decoder = FPDecoder(params)

    def forward(self, x):
        z_samp, z_mean_log_var_output = self.encoder(x)
        x_out = self.decoder(z_samp)
        return x_out, z_mean_log_var_output