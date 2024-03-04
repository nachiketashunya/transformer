import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path 
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# Function to build tokenizer
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[EOS]", "[SOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['src_lang']}-{config['tgt_lang']}", split='train')

    # Build Tokenizers
    src_lang_tokenizer = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tgt_lang_tokenizer = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])

    # Split Dataset - Train:Val = 9:1
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, src_lang_tokenizer, tgt_lang_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, src_lang_tokenizer, tgt_lang_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = src_lang_tokenizer.encode(item['translation'][config['src_lang']]).ids
        tgt_ids = tgt_lang_tokenizer.encode(item['translation'][config['tgt_lang']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of source sentences: {max_len_src}")
    print(f"Max length of target sentences: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader, src_lang_tokenizer, tgt_lang_tokenizer

def get_model(config, src_vocab_len, tgt_vocab_len):
    model = build_transformer(src_vocab_len, tgt_vocab_len, config['seq_len'], config['seq_len'])
    return model 

def greedy_decode(model, source, source_mask, src_tokenizer, tgt_tokenizer, max_len, device):
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    # Precompute the encoder the output
    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
            
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break 
    
    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Console width for displaying
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            if count == num_examples:
                break 
            

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading Model: {model_filename}")

        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (bs, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (bs, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (bs, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (bs, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (bs, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (bs, seq_len, d_model)
            proj_output = model.project(decoder_output) # (bs, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (bs, seq_len)

            # (bs, seq_len, vocab_size) --> (bs * seq_len, vocab_size)
            loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # Save the model 
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    config = get_config()
    train_model(config)
