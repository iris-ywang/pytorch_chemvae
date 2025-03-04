import numpy as np

class PairingDataByFeature():
    def __init__(self, data, pair_ids, mapping, feature_datatype):
        self.data = data
        self.mapping = mapping
        self.feature_datatype = feature_datatype

        self.n_samples, self.n_columns = np.shape(data)
        self.permutation_pairs = pair_ids
        self.n_combinations = len(self.permutation_pairs)

    def parallelised_pairing_process(self, combination_id):

        sample_id_a, sample_id_b = self.permutation_pairs[combination_id]
        sample_a = self.data[sample_id_a: sample_id_a + 1, :]
        sample_b = self.data[sample_id_b: sample_id_b + 1, :]
        pair_ab = self.pair_boolean_or_continuous_features(sample_a, sample_b)

        return (sample_id_a, sample_id_b), pair_ab

    def pair_boolean_or_continuous_features(self, sample_a, sample_b):
        a = sample_a[0, :]
        b = sample_b[0, :]
        new_sample = [a[0] - b[0]]

        for feature_id in range(1, len(a)):
            is_boolean = self.feature_datatype[feature_id]

            if is_boolean:
                feature_combi = (a[feature_id], b[feature_id])
                new_sample.append(self.mapping[feature_combi])
            else:  # is_continuous
                new_sample.append(a[feature_id] - b[feature_id])
                new_sample.append(a[feature_id])
        return new_sample


def pair_by_pair_id_per_feature(data, pair_ids, n_bins_max=2):
    data = np.array(data)
    n_samples, n_columns = data.shape
    feature_datatype = {}
    for feature in range(1, n_columns):
        n_unique = len(np.unique(data[:, feature]))
        if n_unique <= n_bins_max:
            feature_datatype[feature] = 1  # boolean
        else:
            feature_datatype[feature] = 0  # continuous

    mapping = {(0, 1): -1, (1, 0): 1, (0, 0): 0, (1, 1): 2}

    pairing_tool = PairingDataByFeature(data, pair_ids, mapping, feature_datatype)

    results = map(pairing_tool.parallelised_pairing_process, range(pairing_tool.n_combinations))
    return np.array([values for _, values in dict(results).items()])
