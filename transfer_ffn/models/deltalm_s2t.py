#!/usr/bin/env python3

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler, TransformerDecoderScriptable
from fairseq.models.hubert import HubertModel
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,

)
from .transformer_base import PTransformerModelBase
from .options import *
# from .transformer_encoder import HyTransformerEncoderBase



from fairseq.models import MODEL_REGISTRY, ARCH_MODEL_INV_REGISTRY



from fairseq.models.transformer import (
    TransformerModel,
)

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoderBase,
    TransformerEncoderBase,
    
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase,
    TransformerEncoderLayerBase
)
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq import utils
from fairseq.file_io import PathManager
import logging

from .transformer_encoder import PTransformerEncoderBase
from .transformer_decoder import PTransformerDecoderBase
from .prefix_lang_encoder import PrefixEncoder

# from .transformer_encoder import 

# from .deltalm import DeltaLMEncoder
# from .deltalm import DeltaLMDecoder
from .transformer_layer import PTransformerEncoderLayerBase
from fairseq.distributed import fsdp_wrap
from ._adapters import Adapter

import time

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    
)

from ._transformer_encoder_adapter import AdapterTransformerEncoderBase
from ._transformer_decoder_adapter import AdapterTransformerDecoderBase




logger = logging.getLogger(__name__)

def upgrade_state_dict_for_deltalm(args,
    state_dict: Dict[str, Any], pretrained_deltalm_checkpoint: str, is_encoder=True,
) -> Dict[str, Any]:


    # torch.save(prefix_encoder, "tmp.pt")
    # prefix_encoder = torch.load("tmp.pt")
    if not os.path.exists(pretrained_deltalm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_deltalm_checkpoint))




    with open(pretrained_deltalm_checkpoint, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    if not args.resume:

        prefix_encoder = PrefixEncoder(args)
        print(666)
        exit()
        state["encoder.prefix_encoder.embedding.weight"] = prefix_encoder.state_dict()["embedding.weight"]

    # deltalm_state_dict = state["weights"]
    deltalm_state_dict = state["model"]


    # MODIFY_PATH["model"]["encoder.prefix_encoder.embedding.weight"] = PREFIX_MODEL["model"]["encoder.prefix_encoder.embedding.weight"]

    new_deltalm_state_dict = {}
    for key in deltalm_state_dict.keys():

        if is_encoder:
            # if "hubert" in key or "subsampler" in key:
            #     continue
            
            if key.startswith('encoder.') or key.startswith('src_embedding.'):
                new_key = key.replace('encoder.', '')
                new_key = new_key.replace('src_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]
        else:
            if key.startswith('decoder.') or key.startswith('tgt_embedding.'):
                new_key = key.replace('decoder.', '')
                new_key = new_key.replace('tgt_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]
    
    deltalm_state_dict = new_deltalm_state_dict




    
    for key in deltalm_state_dict.keys():
        # print(key)
        map_key = key
        map_key = map_key.replace('.ffn_1.fc1', '.fc3')
        map_key = map_key.replace('.ffn_1.fc2', '.fc4')
        map_key = map_key.replace('.ffn_2', '')
        map_key = map_key.replace('.ffn.', '.')
        map_key = map_key.replace('emb_layer_norm', 'layernorm_embedding')
        # assert map_key in state_dict, map_key



        if 'embed_positions' in key or 'embed_tokens' in key:
        # if 'embed_tokens' in key:
            left_size = state_dict[map_key].size(0)
            right_size = deltalm_state_dict[key].size(0)
            if left_size <= right_size:
                state_dict[map_key] = deltalm_state_dict[key][:left_size]
            else:
                state_dict[map_key][:right_size] = deltalm_state_dict[key]
        else:
            state_dict[map_key] = deltalm_state_dict[key]

    return state_dict

@register_model("deltalm_transformer_adapter_emd")
class DeltalmTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/s2t"
        model_ids = [
            "s2t_transformer_s-en-asr-librispeech",
            "s2t_transformer_m-en-asr-librispeech",
            "s2t_transformer_l-en-asr-librispeech",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            **kwargs,
        )
        return S2THubInterface(x["args"], x["task"], x["models"][0])

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.epoch = 1
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        # hubert arguments
        parser.add_argument(
            "--hubert-model-path",
            type=str,
            metavar="STR",
            help="path/to/hubert/model"
        )
        parser.add_argument(
            "--freeze-hubert",
            action="store_true",
            help="if we want to freeze the hubert features"
        )
        # subsampler arguments
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            help="# of channels in Conv1d subsampling layers",
        )
        # pretrain
        parser.add_argument(
            "--pretrained_deltalm_checkpoint", 
            type=str,
            help="model to take mt encoder/decoder weight from (for initialization)",
        )


        parser.add_argument(
            "--pretrained-deltalm-checkpoint",
            type=str,
            metavar="STR",
        )
        # add 
        parser.add_argument(
            "--prefix_length",
            type=int,
            default=24,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--use_prefix_decoder",
            default='False', 
            type=bool,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )

        parser.add_argument(
            "--prefix_decoder_length",
            type=int,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )

        parser.add_argument(
            "--frozen_plm",default='False', 
            # action='store_true',
            # action='store_true',
            # default=False,
            # type=bool,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )
        # parser.add_argument("--frozen_plm",
        #                 action="store_true",
        #                 help="Run or not.")
        parser.add_argument(
            "--resume",default='False', 
            # action='store_true',
            # action='store_true',
            # default=False,
            type=bool,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )



        # ------------------------------------------------------------------------------------------------
        # args for adapter layers
        # ------------------------------------------------------------------------------------------------
        parser.add_argument('--adapters', default=False, action='store_true',
                            help="Add an adapter layer after each Transformer layer.")
        parser.add_argument('--adapters-multi', default=False, action='store_true',
                            help="Use multiple different adapters, one per task (e.g., language(-pair)).")
        parser.add_argument('--adapters-bottle', type=int, metavar='D',
                            help="The dimensionality of the adapter bottleneck representation.")
        parser.add_argument('--adapters-activation-fn', type=str, default="relu",
                            choices=utils.get_available_activation_fns(),
                            help="Activation function for the adapters bottleneck.")
        parser.add_argument('--adapters-static-layernorm',  default=False, action='store_true',
                            help="Use LayerNorm without trainable parameters in adapter layers.")
        parser.add_argument('--adapters-encoder-lang', default="src", type=str,
                            choices=['src', 'tgt', 'pair'],
                            help='"The key to use for indexing the encoder language adapters."')
        parser.add_argument('--adapters-decoder-lang', default="tgt", type=str,
                            choices=['src', 'tgt', 'pair'],
                            help="The key to use for indexing the decoder language adapters.")
        parser.add_argument('--adapters-efficient', default=False, action='store_true',
                            help="Use a single weight matrix for all adapter layers. "
                                 "Improves distributed training efficiency.")




    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):

        transformer_encoder_config = TransformerConfig.from_namespace(args)


        transformer_encoder_config.adapters_activation_fn = args.adapters_activation_fn
        transformer_encoder_config.adapters_bottle = args.adapters_bottle
        transformer_encoder_config.adapters_static_layernorm =  args.adapters_static_layernorm  


        # config.prefix_length, config.encoder_layers * 2 * config.encoder_embed_dim
        encoder =  DeltalmTransformerEncoder(transformer_encoder_config, task.target_dictionary, embed_tokens)
        if args.frozen_plm:
            for name, parameter in encoder.named_parameters():
                # print(name, parameter.numel())
                if "transformer_layers" in name:
                    parameter.requires_grad = False
                elif "embed_tokens" in name:
                    parameter.requires_grad = False
                elif "embed_positions" in name:
                    parameter.requires_grad = False
                elif "layernorm_embedding" in name:
                    parameter.requires_grad = False          
                # elif "dim_proj" in name:
                #     parameter.requires_grad = True
                #     print("dim_proj", name)
                #     exit()
                else:
                    parameter.requires_grad = True 
      

        total_num = sum(p.numel() for p in encoder.parameters())
        trainable_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print({'Total': total_num, 'Trainable': trainable_num}) 

        # print(encoder)        
        # print(encoder)
        # exit()  
        return encoder



    @classmethod
    def build_decoder(cls, args, task, embed_tokens):

        decoder = DeltaLMDecoder(TransformerConfig.from_namespace(args), task.target_dictionary, embed_tokens)
        if args.frozen_plm:
            for name, parameter in decoder.named_parameters():
                print(name, parameter.numel())
                if "adapters" in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        decoder_total_num = sum(p.numel() for p in decoder.parameters())
        decoder_trainable_num = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print({'Total': decoder_total_num, 'Trainable': decoder_trainable_num})
     
        return decoder
        # return DeltaLMDecoder(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        FT_MODEL="/workspace/s2t/deltalm_data/save_dir/de/pruning_layer_rate_090_finetuned_layer6_dim1/ft_model.pt"
        self.hubert_model_path = FT_MODEL

        ckpt = checkpoint_utils.load_checkpoint_to_cpu(self.hubert_model_path)
        args = ckpt["cfg"]

        task = tasks.setup_task(args.task)
        if "task_state" in ckpt:
            task.load_state_dict(ckpt["task_state"])
        self.ft_model = task.build_model(args.model)
        self.ft_model.load_state_dict(ckpt["model"])    


    # Extract FFN modules from encoder and decoder and create modellists
        self.encoder_ffn = []
        self.decoder_ffn = []
        for layer in self.ft_model.encoder.layers:
            self.encoder_ffn.append(layer.ffn)

        for layer in self.ft_model.decoder.layers:
            self.decoder_ffn.append(layer.ffn)




        encoder_embed_tokens = decoder_embed_tokens
        encoder = cls.build_encoder(args, task, encoder_embed_tokens, self.encoder_ffn)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens,  self.decoder_ffn)

  
        total_num = sum(p.numel() for p in encoder.parameters())
        trainable_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print({'Total': total_num, 'Trainable': trainable_num}) 
        decoder_total_num = sum(p.numel() for p in decoder.parameters())
        decoder_trainable_num = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print({'Total': decoder_total_num + total_num, 'Trainable': decoder_trainable_num + trainable_num})



        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, mode, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, mode=mode)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out



## change deltalm to orig transformer


class DeltaLMDecoder(AdapterTransformerDecoderBase):
    def __init__(self, args, dictionary, embed_tokens, decndoer_ffn, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        if getattr(args, "pretrained_deltalm_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                args,
                state_dict=self.state_dict(),
                pretrained_deltalm_checkpoint=args.pretrained_deltalm_checkpoint,
                is_encoder=False,
            )

            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            logger.info("Load DeltaLM's decoder from {0}".format(args.pretrained_deltalm_checkpoint))


    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = DeltaLMDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer


class DeltalmTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, dictionary=None, embed_tokens=None, return_fc=False):
        super().__init__(None)

        self.num_updates = 0
        self.encoder_layerdrop = args.encoder.layerdrop
        self.return_fc = return_fc
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad()

        # load hubert
        self.hubert_model_path = getattr(args, "hubert_model_path", None)
        self.freeze_hubert = getattr(args, "freeze_hubert", False)
        assert self.hubert_model_path is not None
        ckpt = checkpoint_utils.load_checkpoint_to_cpu(self.hubert_model_path)
        hubert_args = ckpt["cfg"]
        task = tasks.setup_task(hubert_args.task)

        if "task_state" in ckpt:
            task.load_state_dict(ckpt["task_state"])
        self.hubert_model = task.build_model(hubert_args.model)
        self.hubert_model.load_state_dict(ckpt["model"])
        self.hubert_model.remove_pretraining_modules()

        if self.freeze_hubert:
            print(6666)
            exit()
            for param in self.hubert_model.parameters():
                param.requires_grad = False
        
        # speech subsample
        if args.conv_kernel_sizes:
            self.subsampler = Conv1dSubsampler(
                hubert_args.model.encoder_embed_dim,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        else:
            self.subsampler = None
            # self.dim_proj = 

            self.dim_proj = nn.ModuleList([])

            self.dim_proj.append(nn.Linear(hubert_args.model.encoder_embed_dim, args.encoder_embed_dim))

        ## deltalm encoder
        # self.transformer_layers = 
        transformer_encoder_config = TransformerConfig.from_namespace(args)


        transformer_encoder_config.adapters_activation_fn = args.adapters_activation_fn
        transformer_encoder_config.adapters_bottle = args.adapters_bottle
        transformer_encoder_config.adapters_static_layernorm =  args.adapters_static_layernorm  

 
        # config.prefix_length, config.encoder_layers * 2 * config.encoder_embed_dim

        # transformer encoder
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        ) 
        
        ## load_deltalm_encoder
        pretrained_deltalm_checkpoint = args.pretrained_deltalm_checkpoint
        if pretrained_deltalm_checkpoint is not None and Path(pretrained_deltalm_checkpoint).exists():
            with open(pretrained_deltalm_checkpoint, "rb") as f:

                # deltalm_state_dict = torch.load(f, map_location=torch.device("cpu"))["weights"]
                deltalm_state_dict = torch.load(f, map_location=torch.device("cpu"))["model"]


        new_deltalm_state_dict = [{} for _ in range(len(self.transformer_layers))]


        # for index, layer in enumerate(range(len(self.transformer_layers))):

        for key in deltalm_state_dict.keys(): 
            
            if key.startswith(f"encoder.layers."):
                index = int(key.split(".")[2])
                # new_key = key.replace('encoder.', '')
                # new_key = new_key.replace('src_embedding.', '')
                # new_deltalm_state_dict[new_key] = deltalm_state_dict[key]
                map_key = key
                map_key = map_key.replace(f'encoder.layers.{index}.', '')

                map_key = map_key.replace('ffn_1.fc1', 'fc3')
                map_key = map_key.replace('ffn_1.fc2', 'fc4')
                map_key = map_key.replace('ffn_2', '')
                map_key = map_key.replace('ffn.', '')
                map_key = map_key.replace('emb_layer_norm', 'layernorm_embedding')

                # print(map_key, key)

                new_deltalm_state_dict[index][map_key] = deltalm_state_dict[key]



            else:
                pass


        for index, layer in enumerate(range(len(self.transformer_layers))):
            # print(new_deltalm_state_dict[index].keys())
            self.transformer_layers[index].load_state_dict(new_deltalm_state_dict[index], strict=True)
            print("over")




        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None



        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        
        # embedding
        self.embed_tokens = embed_tokens
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_tokens.embedding_dim, export=export)
        else:
            self.layernorm_embedding = None
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, 
            args.encoder_embed_dim,
            self.padding_idx,
        )
        # embedding new 
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder.learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        if args.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=args.export)
   
            layernorm_embedding_dict = {}
            layernorm_embedding_dict["encoder.layernorm_embedding.weight"] = deltalm_state_dict["encoder.layernorm_embedding.weight"]
            layernorm_embedding_dict["encoder.layernorm_embedding.bias"] = deltalm_state_dict["encoder.layernorm_embedding.bias"]
            self.layernorm_embedding.load_state_dict(layernorm_embedding_dict, strict=False)
 
        else:
            self.layernorm_embedding = None

        # deltalm_state_dict
        print(deltalm_state_dict.keys())
        embed_tokens_dict = {}
        embed_tokens_dict["encoder.embed_tokens.weight"] = deltalm_state_dict["encoder.embed_tokens.weight"]
        # embed_tokens_dict["encoder.embed_tokens.bias"] = deltalm_state_dict["encoder.embed_tokens.bias"]     

        embed_positions_dict = {}
        embed_positions_dict["encoder.embed_positions.weight"] = deltalm_state_dict["encoder.embed_positions.weight"]
        # embed_positions_dict["encoder.embed_positions.bias"] = deltalm_state_dict["encoder.embed_positions.bias"]


        self.embed_tokens.load_state_dict(embed_tokens_dict, strict=False)
        self.embed_positions.load_state_dict(embed_positions_dict, strict=False)

    ## end


        self.ffn = nn.ModuleList([])
        for i in range(len(self.transformer_layers)):
            self.ffn.append(Adapter(transformer_encoder_config.encoder.embed_dim,
                                            args.adapters_bottle,
                                            args.adapters_activation_fn,
                                            args.adapters_static_layernorm))


        FT_MODEL=args.external_path
        self.ext_path = FT_MODEL

        ckpt = checkpoint_utils.load_checkpoint_to_cpu(self.ext_path)
        

        model_args = ckpt["cfg"]

        task = tasks.setup_task(model_args.task)
        if "task_state" in ckpt:
            task.load_state_dict(ckpt["task_state"])

        self.ft_model = task.build_model(model_args.model)

        self.ft_model.load_state_dict(ckpt["model"])    
        
        self.ft_model.eval()    
        self.ffn = load_weights(self.ft_model, "encoder")

    def build_encoder_layer(self, cfg):
        layer = TransformerEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


    def _get_hubert_features(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        hubert_args = {
            "source": src_tokens,
            "padding_mask": padding_mask,
            "mask": False,
        }
        x, padding_mask = self.hubert_model.extract_features(**hubert_args)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return x, padding_mask, output_length
    
    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def _forward(self, src_tokens, src_lengths, mode, return_all_hiddens=False):
        if mode == "st":
            x, encoder_padding_mask, input_lengths = self._get_hubert_features(src_tokens, src_lengths)
            if self.subsampler is not None:
                x, input_lengths = self.subsampler(x, input_lengths)
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)
                x = x.transpose(0, 1)  # T x B x C -> B x T x C
            else:
                x = self.dim_proj(x)
            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)
            x = self.dropout_module(x)
        else:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
            x, _ = self.forward_embedding(src_tokens)
            if has_pads:
                x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_embedding = x
        x = x.transpose(0, 1)  # B x T x C -> T x B x C

        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)


        for index, layer in enumerate(self.transformer_layers):
 
            x = layer(x, encoder_padding_mask)
            ## ffn
            x = self.ffn[index](x)

            if return_all_hiddens:
                encoder_states.append(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, mode, return_all_hiddens=False):
        x = self._forward(
            src_tokens, src_lengths, mode, return_all_hiddens=return_all_hiddens
        )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

class DeltaLMDecoderLayer(TransformerDecoderLayerBase):
    
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayerBase, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        src_lang_id = None,
        tgt_lang_id = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

# @register_model_architecture(
#     "deltalm_transformer", "deltalm_base"
# )
# def base_architecture(args):
#     args.encoder_embed_dim = 768
#     args.encoder_ffn_embed_dim = 3072
#     args.encoder_layers = 12
#     args.encoder_attention_heads = 12
#     args.encoder_normalize_before = False
#     args.encoder_learned_pos = True
#     args.decoder_embed_dim = 768
#     args.decoder_ffn_embed_dim = 3072
#     args.decoder_layers = 6
#     args.decoder_attention_heads = 12
#     args.decoder_normalize_before = False
#     args.decoder_learned_pos = True
#     args.activation_fn = "gelu"
#     args.no_scale_embedding = True
#     args.layernorm_embedding = True
#     args.max_positions = 512





@register_model_architecture(model_name="deltalm_transformer_ffn", arch_name="deltalm_transformer_ffn")
def base_architecture(args):
    # subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    # args.no_token_positional_embeddings = getattr(
    #     args, "no_token_positional_embeddings", False
    # )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.activation_fn = "gelu"
    args.no_scale_embedding = True
    args.layernorm_embedding = True
    args.max_positions = 512
    args.encoder_learned_pos = True
    args.encoder_embed_dim = 768
    args.encoder_ffn_embed_dim = 3072
    args.encoder_layers = 12
    args.encoder_attention_heads = 12
    args.encoder_normalize_before = False
    args.encoder_learned_pos = True
    args.decoder_embed_dim = 768
    args.decoder_ffn_embed_dim = 3072
    args.decoder_layers = 6
    args.decoder_attention_heads = 12
    args.decoder_normalize_before = False
    args.decoder_learned_pos = True
    args.activation_fn = "gelu"
    args.no_scale_embedding = True
    args.layernorm_embedding = True
    args.max_positions = 512
# @register_model_architecture(model_name="hubert_transformer", arch_name="hubert_transformer_postln")
# def hubert_transformer_postln(args):
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
#     base_architecture(args)