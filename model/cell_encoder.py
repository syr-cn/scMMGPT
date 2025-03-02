from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import torch
import json

"""Config of scGPT model
hyperparameter_defaults = dict(
    seed=42,
    dataset_name="PBMC_10K",
    do_train=True,
    load_model="save/scGPT_bc",
    mask_ratio=0.4,
    epochs=30,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    lr=1e-4,
    batch_size=64,
    layer_size=128,
    nlayers=4,
    nhead=4,
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
)
"""

class CellEncoder(torch.nn.Module):
    def __init__(self, vocab_path, model_path, model_config_path):
        super(CellEncoder, self).__init__()
        self.vocab = GeneVocab.from_file(vocab_path)
        self.vocab.set_default_index(self.vocab["<pad>"])

        with open(model_config_path, "r") as f:
            model_configs = json.load(f)
        self.d_model = model_configs["embsize"]
        self.nhead = model_configs["nheads"]
        self.d_hid = model_configs["d_hid"]
        self.nlayers = model_configs["nlayers"]
        self.nlayers_cls = model_configs['n_layers_cls']

        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.d_model,
            nhead=self.nhead,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            nlayers_cls=self.nlayers_cls,
            vocab=self.vocab,
            dropout=0.2,
            # pad_token="<pad>",
            # pad_value=-2,
            # do_mvc=False,
            # do_dab=True,
            # use_batch_labels=False,
            # domain_spec_batchnorm=False,
            # n_input_bins=51,
            # ecs_threshold=0.8,
            # explicit_zero_prob=True,
            # use_fast_transformer=True,
            # pre_norm=False,
            n_cls=1,
            pad_token="<pad>",
            pad_value=-2,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style="continuous",
            n_input_bins=51,
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            use_fast_transformer=True,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        del self.model.cls_decoder
        missing_keys, unexpected_keys = self.model.load_state_dict(torch.load(model_path), strict=False)
        print(f'Loaded scGPT model from {model_path}')
        if len(missing_keys) or len(unexpected_keys):
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)
        self.num_features = model_configs["embsize"]

    def forward(self, batch_data):
        input_gene_ids = batch_data["gene_ids"].to(self.get_device())
        input_values = batch_data["values"].to(self.get_device())
        src_key_padding_mask = batch_data['padding_mask'].to(self.get_device())
        
        transformer_output = self.model._encode(input_gene_ids, input_values, src_key_padding_mask) # [batch_size, seq_len, d_model]
        return transformer_output

    def get_device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CellEncoder")
        parser.add_argument('--vocab_path', type=str, default='scgpt_pretrained/human/vocab.json')
        parser.add_argument('--model_path', type=str, default='scgpt_pretrained/human/best_model.pt')
        parser.add_argument('--model_config_path', type=str, default='scgpt_pretrained/human/args.json')
        return parent_parser