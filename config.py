from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "epochs": 20,
        "lr": 0.0001,
        "seq_len": 350,
        "src_lang": "en",
        "tgt_lang": "it",
        "d_model": 512,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch = 0, latest = False):
    model_folder = config['model_folder']
   
    if latest:
        model_filename = f"{config['model_basename']}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        
        if len(weights_files) == 0:
            return None
        
        weights_files.sort()

        return str(weights_files[-1])
    else:
        model_basename = config['model_basename']
        model_filename = f"{model_basename}{epoch}.pt"
        
        return str(Path(".") / model_folder / model_filename)
