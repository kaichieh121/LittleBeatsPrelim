import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from LittleBeatsPrelim.wav2vec_scripts.load_w2v_from_fairseq_weights import load_fairseq_weights

from .modeling_wav2vec2_custom import Wav2Vec2PreTrainedModel as Wav2Vec2PreTrainedModel_custom
from .modeling_wav2vec2_custom import Wav2Vec2Model as Wav2Vec2Model_custom
from .modeling_wav2vec2_custom import Wav2Vec2ForPreTraining

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from .wav2vec_audio_ecg_model import AudioEcgThreeLayerClassifier, ImuClassifier, AllThreeLayerClassifier, AudioTwoLayerClassifier

from LittleBeatsPrelim.baseline_scripts.baseline_ecg import BaselineEcg
from LittleBeatsPrelim.baseline_scripts.baseline_resp_ecg import BaselineRespEcg

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def create_model(config, embedding_type, lb_audio_pretrained_weights, bp_ecg_pretrained_weights):
    mode = config.mode
    if mode == 'mono':
        model = OneModalityModel(config=config, mode=embedding_type)
        if embedding_type == 0:
            model = load_fairseq_weights(model, 'wav2vec2', lb_audio_pretrained_weights['model'], config)
        else:
            model = load_fairseq_weights(model, 'wav2vec2', bp_ecg_pretrained_weights['model'], config)
        # model = OneModalityModel.from_pretrained(model_name_or_path, config=config, mode=embedding_type)
    elif mode == 'stereo':
        model = TwoModalityModel.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config)
        # model = TwoModalityModel(config)
    elif mode == 'triple' or mode == 'stereo+limu':
        if config.pretrain:
            model = AllModalityModelForPreTraining(config=config, embedding_type=embedding_type)
        else:
            model = AllModalityModel(config=config, embedding_type=embedding_type)
            model = load_fairseq_weights(model, 'wav2vec2', lb_audio_pretrained_weights['model'], config)
            model = load_fairseq_weights(model, 'wav2vec2_copy', bp_ecg_pretrained_weights['model'], config)
    elif mode == 'baseline_ecg' or mode == 'baseline_resp_ecg':
        model = BaselineModel(config=config, mode=mode, embedding_type=1)
    model.update_weights()
    return model

class AllModalityModelForPreTraining(Wav2Vec2PreTrainedModel_custom):
    def __init__(self, config, embedding_type):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2 = Wav2Vec2ForPreTraining(config)
        self.embedding_type = embedding_type
        self.mode = config.mode
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.BoolTensor] = None,
            sampled_negative_indices: Optional[torch.BoolTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output = self.wav2vec2(input_values=input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        return output

    def update_weights(self):
        pass



class AllModalityModel(Wav2Vec2PreTrainedModel_custom):
    def __init__(self, config, embedding_type):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.wav2vec2_copy = Wav2Vec2Model(config)
        self.wav2vec2_stereo = Wav2Vec2Model_custom(config)
        self.classifier = AllThreeLayerClassifier(config)

        self.embedding_type = embedding_type
        self.mode = config.mode


        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def update_weights(self):
        # self.feature_extractor
        for i in range(self.config.num_feat_extract_layers):
            self.wav2vec2_stereo.feature_extractor.conv_layers[i].conv1.load_state_dict(self.wav2vec2.feature_extractor.conv_layers[i].conv.state_dict())
            self.wav2vec2_stereo.feature_extractor.conv_layers[i].conv2.load_state_dict(self.wav2vec2_copy.feature_extractor.conv_layers[i].conv.state_dict())
            if i == 0:
                self.wav2vec2_stereo.feature_extractor.conv_layers[i].layer_norm1.load_state_dict(self.wav2vec2.feature_extractor.conv_layers[i].layer_norm.state_dict())
                self.wav2vec2_stereo.feature_extractor.conv_layers[i].layer_norm2.load_state_dict(self.wav2vec2_copy.feature_extractor.conv_layers[i].layer_norm.state_dict())


        # self.feature_projection 1 and 2
        self.wav2vec2_stereo.feature_projection1.load_state_dict(self.wav2vec2.feature_projection.state_dict())
        self.wav2vec2_stereo.feature_projection2.load_state_dict(self.wav2vec2_copy.feature_projection.state_dict())

        # # self.encoder
        self.wav2vec2_stereo.encoder.pos_conv_embed1.load_state_dict(self.wav2vec2.encoder.pos_conv_embed.state_dict())
        self.wav2vec2_stereo.encoder.layer_norm1.load_state_dict(self.wav2vec2.encoder.layer_norm.state_dict())
        self.wav2vec2_stereo.encoder.pos_conv_embed2.load_state_dict(self.wav2vec2_copy.encoder.pos_conv_embed.state_dict())
        self.wav2vec2_stereo.encoder.layer_norm2.load_state_dict(self.wav2vec2_copy.encoder.layer_norm.state_dict())
        # for i in range(self.config.num_hidden_layers):
        #     self.wav2vec2_stereo.encoder.layers[i].attention1.k_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.k_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention1.v_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.v_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention1.q_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.q_proj.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].attention1.out_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.out_proj.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].layer_norm1.load_state_dict(self.wav2vec2.encoder.layers[i].layer_norm.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].feed_forward1.load_state_dict(self.wav2vec2.encoder.layers[i].feed_forward.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].final_layer_normrm1.load_state_dict(self.wav2vec2.encoder.layers[i].final_layer_norm.state_dict())

            # self.wav2vec2_stereo.encoder.layers[i].attention2.k_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.k_proj.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].attention2.v_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.v_proj.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].attention2.q_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.q_proj.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].attention2.out_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.out_proj.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].layer_norm2.load_state_dict(self.wav2vec2_copy.encoder.layers[i].layer_norm.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].feed_forward2.load_state_dict(self.wav2vec2_copy.encoder.layers[i].feed_forward.state_dict())
            # self.wav2vec2_stereo.encoder.layers[i].final_layer_norm2.load_state_dict(self.wav2vec2_copy.encoder.layers[i].final_layer_norm.state_dict())

        # self.wav2vec2_stereo.encoder.layer_norm1.load_state_dict(self.wav2vec2.encoder.layer_norm.state_dict())
        # self.wav2vec2_stereo.encoder.layer_norm2.load_state_dict(self.wav2vec2_copy.encoder.layer_norm.state_dict())
        del(self.wav2vec2)
        del(self.wav2vec2_copy)
        self.wav2vec2 = self.wav2vec2_stereo
        del(self.wav2vec2_stereo)

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
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state[0]
        attentions = outputs.extract_features
        limu_hidden_states = outputs.last_hidden_state[1]
        limu_hidden_states = self.merged_strategy(limu_hidden_states, mode=self.pooling_mode)
        hidden_states1 = self.merged_strategy(hidden_states[:, 0, :, :], mode=self.pooling_mode)
        hidden_states2 = self.merged_strategy(hidden_states[:, 1, :, :], mode=self.pooling_mode)
        # for early fusion
        ##################
        # limu_hidden_states = self.merged_strategy(limu_hidden_states, mode=self.pooling_mode)
        ##################
        if self.config.mode == 'stereo+limu':
            hidden_states = torch.cat((hidden_states1, hidden_states2, input_values[:,self.embedding_type,:72]), dim=1)
        elif self.config.mode == 'triple':
            hidden_states = torch.cat((hidden_states1, hidden_states2, limu_hidden_states), dim=1)

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

class TwoModalityModel(Wav2Vec2PreTrainedModel_custom):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.wav2vec2_copy = Wav2Vec2Model(config)
        self.wav2vec2_stereo = Wav2Vec2Model_custom(config)
        self.classifier = AudioEcgThreeLayerClassifier(config)

        self.init_weights()

    def update_weights(self):
        self.wav2vec2_copy.load_state_dict(self.wav2vec2.state_dict())
        # # self.feature_extractor
        # for i in range(self.config.num_feat_extract_layers):
        #     self.wav2vec2_stereo.feature_extractor.conv_layers[i].conv1.load_state_dict(self.wav2vec2.feature_extractor.conv_layers[i].conv.state_dict())
        #     self.wav2vec2_stereo.feature_extractor.conv_layers[i].layer_norm1.load_state_dict(self.wav2vec2.feature_extractor.conv_layers[i].layer_norm.state_dict())
        #     self.wav2vec2_stereo.feature_extractor.conv_layers[i].conv2.load_state_dict(self.wav2vec2_copy.feature_extractor.conv_layers[i].conv.state_dict())
        #     self.wav2vec2_stereo.feature_extractor.conv_layers[i].layer_norm2.load_state_dict(self.wav2vec2_copy.feature_extractor.conv_layers[i].layer_norm.state_dict())
        #
        # # self.feature_projection 1 and 2
        # self.wav2vec2_stereo.feature_projection1.load_state_dict(self.wav2vec2.feature_projection.state_dict())
        # self.wav2vec2_stereo.feature_projection2.load_state_dict(self.wav2vec2_copy.feature_projection.state_dict())
        #
        # # self.encoder
        # self.wav2vec2_stereo.encoder.pos_conv_embed1.load_state_dict(self.wav2vec2.encoder.pos_conv_embed.state_dict())
        # self.wav2vec2_stereo.encoder.pos_conv_embed2.load_state_dict(self.wav2vec2_copy.encoder.pos_conv_embed.state_dict())
        # self.wav2vec2_stereo.encoder.layer_norm1.load_state_dict(self.wav2vec2.encoder.layer_norm.state_dict())
        # self.wav2vec2_stereo.encoder.layer_norm2.load_state_dict(self.wav2vec2_copy.encoder.layer_norm.state_dict())
        # for i in range(self.config.num_hidden_layers):
        #     self.wav2vec2_stereo.encoder.layers[i].attention1.k_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.k_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention1.v_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.v_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention1.q_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.q_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention1.out_proj.load_state_dict(self.wav2vec2.encoder.layers[i].attention.out_proj.state_dict())
        #
        #     self.wav2vec2_stereo.encoder.layers[i].attention2.k_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.k_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention2.v_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.v_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention2.q_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.q_proj.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].attention2.out_proj.load_state_dict(self.wav2vec2_copy.encoder.layers[i].attention.out_proj.state_dict())
        #
        #     self.wav2vec2_stereo.encoder.layers[i].layer_norm1.load_state_dict(self.wav2vec2.encoder.layers[i].layer_norm.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].layer_norm2.load_state_dict(self.wav2vec2_copy.encoder.layers[i].layer_norm.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].feed_forward1.load_state_dict(self.wav2vec2.encoder.layers[i].feed_forward.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].feed_forward2.load_state_dict(self.wav2vec2_copy.encoder.layers[i].feed_forward.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].final_layer_norm1.load_state_dict(self.wav2vec2.encoder.layers[i].final_layer_norm.state_dict())
        #     self.wav2vec2_stereo.encoder.layers[i].final_layer_norm2.load_state_dict(self.wav2vec2_copy.encoder.layers[i].final_layer_norm.state_dict())
        del (self.wav2vec2)
        del (self.wav2vec2_copy)
        self.wav2vec2 = self.wav2vec2_stereo
        del (self.wav2vec2_stereo)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state[0]
        attentions = outputs.extract_features
        hidden_states1 = self.merged_strategy(hidden_states[:,0,:,:], mode=self.pooling_mode)
        hidden_states2 = self.merged_strategy(hidden_states[:,1,:,:], mode=self.pooling_mode)
        hidden_states = torch.cat((hidden_states1, hidden_states2), dim=1)

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

class OneModalityModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config, mode):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2_audio = Wav2Vec2Model(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = AudioTwoLayerClassifier(config)
        self.imu_classifier = ImuClassifier()
        self.init_weights()
        self.mode = mode

    def update_weights(self):
        self.wav2vec2_audio = self.wav2vec2
        del(self.wav2vec2)

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
            outputs = self.wav2vec2_audio(
                input_values[:, self.mode, :],
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


class BaselineModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config, mode, embedding_type):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.mode = mode
        self.embedding_type = embedding_type
        # self.wav2vec2 = Wav2Vec2Model(config)

        if mode == "baseline_ecg":
            self.baseline_model = BaselineEcg()
        if mode == "baseline_resp_ecg":
            self.baseline_model = BaselineRespEcg()
        self.init_weights()


    def update_weights(self):
        # del(self.wav2vec2)
        pass

    def freeze_feature_extractor(self):
        pass

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

        if self.mode == "baseline_resp_ecg":
            logits = self.baseline_model(input_values[:, 0, :], input_values[:, 1, :])
            hidden_states = None
            attentions = None
        elif self.mode == "baseline_ecg":
            logits = self.baseline_model(input_values[:, 1, :])
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


        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )