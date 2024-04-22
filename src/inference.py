from architectures.machine_translation_transformer import MachineTranslationTransformer
from tokenizers import Tokenizer
import torch
from validation import dataset1, dataset2
import pandas as pd
import os

# Initialize configuration
import wandb
from config import configs
config_name='unofficial_single_gpu_config' # MODIFY THIS TO CHANGE CONFIGURATION
wandb.init(config=configs[config_name],project="attention-is-all-you-need-paper", entity="bkoch4142")


class InferenceApp:
    def __init__(self):
        # Device handling
        if wandb.config.DEVICE=='gpu':
            if not torch.cuda.is_available():
                raise ValueError('GPU is not available.')
            self.device = 'cuda'
        else:
            self.device='cpu'

    def main(self):

        model = MachineTranslationTransformer(
            d_model=wandb.config.D_MODEL,
            n_blocks=wandb.config.N_BLOCKS,
            src_vocab_size=wandb.config.VOCAB_SIZE,
            trg_vocab_size=wandb.config.VOCAB_SIZE,
            n_heads=wandb.config.N_HEADS,
            d_ff=wandb.config.D_FF,
            dropout_proba=wandb.config.DROPOUT_PROBA
        )

        if wandb.config.PRETRAINED_MODEL_PTH:
            if self.device == 'GPU':
                if torch.cuda.is_available():
                    model.load_state_dict(
                        torch.load(wandb.config.PRETRAINED_MODEL_PTH, map_location=torch.device('cuda')))
                else:
                    model.load_state_dict(
                        torch.load(wandb.config.PRETRAINED_MODEL_PTH, map_location=torch.device('mps')))
            else:
                model.load_state_dict(torch.load(wandb.config.PRETRAINED_MODEL_PTH, map_location=torch.device('cpu')))
                print(f"Loaded model from {wandb.config.PRETRAINED_MODEL_PTH}")

        model.eval()

        tokenizer_pth = os.path.join(wandb.config.RUNS_FOLDER_PTH, wandb.config.RUN_NAME, 'tokenizer.json')
        tokenizer = Tokenizer.from_file(tokenizer_pth)

        en1 = dataset1['English']
        de1 = []
        for en in en1:
            de = model.translate(en, tokenizer)
            de1.append(de)
        dataset1.update({'Predicted German': de1})
        df1_out = pd.DataFrame(dataset1)
        df1_out_pth = os.path.join(wandb.config.RUNS_FOLDER_PTH, wandb.config.RUN_NAME, 'df1.csv')
        df1_out.to_csv(df1_out_pth)

        en2 = dataset2['English']
        de2 = []
        for en in en2:
            de = model.translate(en, tokenizer)
            de2.append(de)
        dataset2.update({'Predicted German': de2})
        df2_out = pd.DataFrame(dataset2)
        df2_out_pth = os.path.join(wandb.config.RUNS_FOLDER_PTH, wandb.config.RUN_NAME, 'df2.csv')
        df2_out.to_csv(df2_out_pth)

        return

if __name__ == "__main__":
    InferenceApp().main()