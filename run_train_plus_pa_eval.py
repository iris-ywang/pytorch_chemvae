from train_vae import run_train_vae
from eval_pa_chembl import run_eval

if __name__ == "__main__":
    exp_file_path = "./trained_models/chembl4016/exp.json"
    n_eval_size = 200
    specific_dataset_path = None
    specific_model_path = None

    run_train_vae(exp_file_path=exp_file_path)
    run_eval(
        exp_file_path=exp_file_path,
        n_test_size=n_eval_size,
        specific_dataset_path=specific_dataset_path,
        specific_model_path=specific_model_path,
    )