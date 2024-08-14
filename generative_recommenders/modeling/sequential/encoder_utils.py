# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gin
from modeling.sequential.embedding_modules import EmbeddingModule
from modeling.sequential.hstu import HSTU
from modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from modeling.sequential.output_postprocessors import OutputPostprocessorModule
from modeling.sequential.sasrec import SASRec

from modeling.similarity_module import GeneralizedInteractionModule, InteractionModule


@gin.configurable
def sasrec_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    ffn_hidden_dim: int = 64,
    ffn_activation_fn: str = "relu",
    ffn_dropout_rate: float = 0.2,
    num_blocks: int = 2,
    num_heads: int = 1,
) -> GeneralizedInteractionModule:
    return SASRec(
        embedding_module=embedding_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_activation_fn=ffn_activation_fn,
        ffn_dropout_rate=ffn_dropout_rate,
        num_blocks=num_blocks,
        num_heads=num_heads,
        similarity_module=interaction_module,
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        activation_checkpoint=activation_checkpoint,
        verbose=verbose,
    )


@gin.configurable
def hstu_encoder(
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    activation_checkpoint: bool,
    verbose: bool,
    num_blocks: int = 2,
    num_heads: int = 1,
    dqk: int = 64,
    dv: int = 64,
    linear_dropout_rate: float = 0.0,
    attn_dropout_rate: float = 0.0,
    normalization: str = "rel_bias",
    linear_config: str = "uvqk",
    linear_activation: str = "silu",
    concat_ua: bool = False,
    enable_relative_attention_bias: bool = True,
) -> GeneralizedInteractionModule:
    return HSTU(
        embedding_module=embedding_module,
        similarity_module=interaction_module,
        input_features_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        max_sequence_len=max_sequence_length,
        max_output_len=max_output_length,
        embedding_dim=embedding_module.item_embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        attention_dim=dqk,
        linear_dim=dv,
        linear_dropout_rate=linear_dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        linear_config=linear_config,
        linear_activation=linear_activation,
        normalization=normalization,
        concat_ua=concat_ua,
        enable_relative_attention_bias=enable_relative_attention_bias,
        verbose=verbose,
    )


@gin.configurable
def get_sequential_encoder(
    module_type: str,
    max_sequence_length: int,
    max_output_length: int,
    embedding_module: EmbeddingModule,
    interaction_module: InteractionModule,
    input_preproc_module: InputFeaturesPreprocessorModule,
    output_postproc_module: OutputPostprocessorModule,
    verbose: bool,
    activation_checkpoint: bool = False,
) -> GeneralizedInteractionModule:
    if module_type == "SASRec":
        model = sasrec_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    elif module_type == "HSTU":
        model = hstu_encoder(
            max_sequence_length=max_sequence_length,
            max_output_length=max_output_length,
            embedding_module=embedding_module,
            interaction_module=interaction_module,
            input_preproc_module=input_preproc_module,
            output_postproc_module=output_postproc_module,
            activation_checkpoint=activation_checkpoint,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported module_type {module_type}")
    return model
