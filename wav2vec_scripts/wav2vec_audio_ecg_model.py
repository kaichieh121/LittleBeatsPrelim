import torch
import math
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    Wav2Vec2Attention,
)

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attention_audio = Wav2Vec2Attention(embed_dim = config.output_hidden_size,
                                                       num_heads = config.num_attention_heads,
                                                       dropout = 0.0,
                                                       is_decoder=False,)
        self.cross_attention_ecg = Wav2Vec2Attention(embed_dim=config.output_hidden_size,
                                                     num_heads=config.num_attention_heads,
                                                     dropout=0.0,
                                                     is_decoder=False,)
    '''
    Wav2Vec2Attention forward definition
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    )
    '''
    def forward(self, audio_features, ecg_features):
        attn_output_audio, attn_weights_reshaped, past_key_value = self.cross_attention_audio(hidden_states=audio_features, key_value_states=ecg_features)
        attn_output_ecg, attn_weights_reshaped, past_key_value = self.cross_attention_ecg(hidden_states=ecg_features, key_value_states=audio_features)
        attn_output_audio = attn_output_audio.squeeze()
        attn_output_ecg = attn_output_ecg.squeeze()
        return attn_output_audio, attn_output_ecg

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class AudioTwoLayerClassifier(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AudioEcgTwoLayerClassifier(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size*2, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AudioEcgThreeLayerClassifier(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ImuClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(72, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class AllThreeLayerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.imu_embedding_size = 72
        self.input_size = config.hidden_size*2+self.imu_embedding_size
        self.l1 = nn.Linear(self.input_size, self.input_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.input_size, self.input_size//2)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(self.input_size//2, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class W2vAvgLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(25, 1, 1)

    def forward(self, x):
        N, B, I, J = x.shape
        x = x.reshape(B, N, I*J)
        x = self.conv(x)
        x = x.reshape(B, I, J)
        return x



class AllModalityModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config, mode):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2_audio = Wav2Vec2Model(config)
        self.wav2vec2_ecg = Wav2Vec2Model(config)
        self.classifier = AllThreeLayerClassifier(config)
        self.mode = mode

        # self.w2v_avg_layer_audio = W2vAvgLayer()
        # self.w2v_avg_layer_ecg = W2vAvgLayer()

        # self.crossattn = CrossAttention(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2_audio.feature_extractor._freeze_parameters()
        self.wav2vec2_ecg.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        # output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2_audio(
            input_values[:,0,:],
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        # hidden_states = torch.mean(torch.stack(outputs.hidden_states), dim=0)
        # hidden_states = self.w2v_avg_layer_audio(torch.stack(outputs.hidden_states))
        attentions_audio = outputs[1]
        hidden_states_audio = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        outputs = self.wav2vec2_ecg(
            input_values[:,1,:],
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        # hidden_states = torch.mean(torch.stack(outputs.hidden_states), dim=0)
        # hidden_states = self.w2v_avg_layer_ecg(torch.stack(outputs.hidden_states))
        attentions_ecg = outputs[1]
        hidden_states_ecg = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        # concatenation after cross attention
        # hidden_states_audio, hidden_states_ecg = self.crossattn(hidden_states_audio.unsqueeze(dim=1), hidden_states_ecg.unsqueeze(dim=1))
        hidden_states = torch.cat((hidden_states_audio,  hidden_states_ecg, input_values[:,self.mode,:72]), dim=1)
        attentions = torch.cat((attentions_audio, attentions_ecg), dim=1)


        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class TwoModalityModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2_audio = Wav2Vec2Model(config)
        self.wav2vec2_ecg = Wav2Vec2Model(config)
        self.classifier = AudioEcgThreeLayerClassifier(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2_audio.feature_extractor._freeze_parameters()
        self.wav2vec2_ecg.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2_audio(
            input_values[:,0,:],
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        attentions_audio = outputs[1]
        hidden_states_audio = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        outputs = self.wav2vec2_ecg(
            input_values[:,1,:],
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        attentions_ecg = outputs[1]
        hidden_states_ecg = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        # concatenation after cross attention
        # hidden_states_audio, hidden_states_ecg = self.crossattn(hidden_states_audio.unsqueeze(dim=1), hidden_states_ecg.unsqueeze(dim=1))
        hidden_states = torch.cat((hidden_states_audio,  hidden_states_ecg), dim=1)
        attentions = torch.cat((attentions_audio, attentions_ecg), dim=1)


        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs[1],
        )

class OneModalityModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config, mode):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2_audio = Wav2Vec2Model(config)
        self.classifier = AudioTwoLayerClassifier(config)
        self.imu_classifier = ImuClassifier()
        self.init_weights()
        self.mode = mode

    def freeze_feature_extractor(self):
        self.wav2vec2_audio.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if embedding is audio or ecg
        if(self.mode < 2):
            if(self.mode == 0):
                outputs = self.wav2vec2_audio(
                    input_values[:,0,:],
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif(self.mode == 1):
                outputs = self.wav2vec2_audio(
                    input_values[:, 1, :],
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            hidden_states = outputs[0]
            attentions = outputs[1]
            hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
            logits = self.classifier(hidden_states)
        # else if embedding is imu limu-bert embedding
        else:
            logits = self.imu_classifier(input_values[:,self.mode,:72])
            hidden_states = None
            attentions = None

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

