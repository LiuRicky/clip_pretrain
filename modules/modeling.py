from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, concat_all_gather
from modules.module_cross import CrossConfig, SpatialAggregationTransformer, TemporalTransformer, Predictor

from modules.module_clip import CLIP, convert_weights

logger = logging.getLogger(__name__)
allgather = AllGather.apply


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["temporal_transformer.frame_position_embeddings.weight"] = val.clone()
                        # state_dict["spatial_aggregator.spatial_selected_tokens"] = val[:4].clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        # if num_layer < 2:  # hard code for spatial aggregator
                        #     state_dict[key.replace("transformer.", "spatial_aggregator.")] = val.clone()
                        #     if key.find("ln_1") > 0:
                        #         state_dict[key.replace("transformer.",
                        #                                "spatial_aggregator.").replace("ln_1", "ln_1_q")] = val.clone()
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "temporal_transformer.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        if task_config.loss_type == 'mom':
            model.copy_params()
        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self.loose_type = False
        if check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))

        cross_config.max_position_embeddings = context_length

        self.loss_type = task_config.loss_type
        self.tempsimsiam = False
        self.mmsimsiam = False
        self.mom = False
        if self.sim_header == "seqTransf":
            self.temporal_transformer = TemporalTransformer(width=transformer_width,
                                                            layers=self.task_config.cross_num_hidden_layers,
                                                            heads=transformer_heads,
                                                            max_frames=context_length)
            if self.tempsimsiam:
                self.temporal_predictor = Predictor(width=transformer_width)
            # self.spatial_aggregator = SpatialAggregationTransformer(width=transformer_width,
            #                                                         layers=2,
            #                                                         heads=transformer_heads)

        if self.mmsimsiam:
            self.video_predictor = Predictor(width=transformer_width)
            self.text_predictor = Predictor(width=transformer_width)

        if self.mom:
            # momentum encoder
            self.clip_m = copy.deepcopy(self.clip)
            self.temporal_transformer_m = copy.deepcopy(self.temporal_transformer)
            self.model_pairs = [[self.clip, self.clip_m],
                                [self.temporal_transformer, self.temporal_transformer_m],
                                ]
            # create the queue
            self.queue_size = task_config.queue_size
            self.momentum = task_config.momentum
            self.alpha = task_config.alpha
            self.register_buffer("text_queue", torch.randn((self.queue_size, 1, embed_dim)))
            self.register_buffer("video_mask_queue",
                                 torch.ones((self.queue_size, task_config.max_frames), dtype=torch.long))
            self.register_buffer("video_queue", torch.randn((self.queue_size, task_config.max_frames, embed_dim)))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.apply(self.init_weights)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient
        logger.info("Copy params from module to momentum modules done")

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, video_feat, video_mask, text_feat):
        # gather keys before updating queue
        video_feat, video_mask, text_feat = video_feat.contiguous(), video_mask.contiguous(), text_feat.contiguous()
        video_feats = concat_all_gather(video_feat)
        video_masks = concat_all_gather(video_mask)
        text_feats = concat_all_gather(text_feat)

        batch_size = video_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.video_queue[ptr:ptr + batch_size, :, :] = video_feats
        self.video_mask_queue[ptr:ptr + batch_size, :] = video_masks
        self.text_queue[ptr:ptr + batch_size, :, :] = text_feats
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True,
                                                                         video_frame=video_frame)

        if self.mom:
            # get momentum features
            with torch.no_grad():
                self._momentum_update()
                sequence_output_m, visual_output_m = self.get_sequence_visual_output_m(input_ids, token_type_ids,
                                                                                       attention_mask,
                                                                                       video, video_mask, shaped=True,
                                                                                       video_frame=video_frame)

        if self.training:
            # There is simsiam in training
            visual_output1, visual_output2, visual_output_temporal1, visual_output_temporal2  = visual_output
            loss = 0.
            if self.loss_type == 'itc':
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output1, attention_mask, video_mask,
                                                        shaped=True, loose_type=self.loose_type)
                sim_loss1 = -torch.diag(F.log_softmax(sim_matrix, dim=-1)).mean()
                sim_loss2 = -torch.diag(F.log_softmax(sim_matrix.T, dim=-1)).mean()
                sim_loss = (sim_loss1 + sim_loss2) / 2
                loss += sim_loss

                # sim_matrix = self.get_similarity_logits(sequence_output, visual_output2, attention_mask, video_mask,
                #                                         shaped=True, loose_type=self.loose_type)
                # sim_loss1 = -torch.diag(F.log_softmax(sim_matrix, dim=-1)).mean()
                # sim_loss2 = -torch.diag(F.log_softmax(sim_matrix.T, dim=-1)).mean()
                # sim_loss = (sim_loss1 + sim_loss2) / 2
                # loss += sim_loss

            if self.tempsimsiam:
                # temporal simsiam loss
                p1, p2, z1, z2 = self.get_temporal_simsiam(visual_output_temporal1, visual_output_temporal2, video_mask)
                # simsiam_loss = (torch.sigmoid(-F.cosine_similarity(p1, z2).mean()) + torch.sigmoid(-F.cosine_similarity(p2, z1).mean())) / 2
                # loss += simsiam_loss
                temp_simsiam_loss1 = -torch.diag(F.log_softmax(p1 @ z2.T, dim=-1)).mean()
                temp_simsiam_loss2 = -torch.diag(F.log_softmax(p2 @ z1.T, dim=-1)).mean()
                loss += (temp_simsiam_loss1 + temp_simsiam_loss2) / 2
                # simsiam_loss = - (torch.log(F.cosine_similarity(p1, z2).mean()) + torch.log(F.cosine_similarity(p2, z1).mean())) / 2
                # simsiam_loss = - (F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean()) / 2
                # loss += simsiam_loss

            if self.mmsimsiam:
                # multimodal simsiam loss
                sim_t2v_simsiam, sim_v2t_simsiam = self.get_mm_simsiam(sequence_output, visual_output1, video_mask)
                # mm_simsiam_loss = (torch.sigmoid(-torch.diag(sim_t2v_simsiam).mean()) + torch.sigmoid(-torch.diag(sim_v2t_simsiam).mean())) / 2
                # loss += mm_simsiam_loss
                simsiam_loss1 = -torch.diag(F.log_softmax(sim_t2v_simsiam, dim=-1)).mean()
                simsiam_loss2 = -torch.diag(F.log_softmax(sim_v2t_simsiam, dim=-1)).mean()
                loss += (simsiam_loss1 + simsiam_loss2) / 2

            if self.mom:
                loss += self.get_momentum_loss(sequence_output, visual_output, video_mask,
                                               sequence_output_m, visual_output_m)
                self._dequeue_and_enqueue(visual_output_m, video_mask, sequence_output_m)

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def encode_temporal(self, visual_output, video_mask):
        if self.sim_header == "meanP":
            # Default: Parameter-free type
            visual_output = visual_output[:, :, 0, :]
        elif self.sim_header == "seqTransf":
            # visual_output shape is (B, T, L, D)
            # visual_mask shape is (B, T)
            # Sequential type: Transformer Encoder
            visual_output_temporal = self.temporal_transformer(visual_output, video_mask)
            visual_output = visual_output_temporal + visual_output
            visual_output = visual_output[:, :, 0, :]
            visual_output_temporal = visual_output_temporal[:, :, 0, :]

        return visual_output, visual_output_temporal

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()  # shape=(B*T,L,D)
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-2),
                                           visual_hidden.size(-1))  # shape=(B,T,L,D)

        # select some spatial tokens
        select_num = 1
        if self.tempsimsiam:
            select_num = visual_hidden.shape[2] // 10

        if hasattr(self, 'spatial_aggregator'):
            B, T, L, D = visual_hidden.shape
            x_cls = visual_hidden[:, :, :1, :]
            x_s = self.spatial_aggregator(visual_hidden[:, :, 1:, :].view(B * T, L - 1, D))
            x_s = x_s.view(B, T, -1, D)
            visual_hidden = torch.cat([x_cls, x_s], dim=2)
            visual_hidden, visual_hidden_temporal = self.encode_temporal(visual_hidden, video_mask)  # shape=(B,T,D)
            return visual_hidden, visual_hidden_temporal
        else:
            if self.training:
                # random select
                random_idx = torch.randperm(visual_hidden.size(2) - 1)
                select_idx1 = random_idx[:select_num-1] + 1
                select_idx2 = random_idx[select_num: select_num*2-1] + 1
                select_idx1 = torch.cat([torch.zeros([1], dtype=torch.long), select_idx1], dim=0)
                select_idx2 = torch.cat([torch.zeros([1], dtype=torch.long), select_idx2], dim=0)
                visual_hidden1 = visual_hidden[:, :, select_idx1, :]
                visual_hidden2 = visual_hidden[:, :, select_idx2, :]

                visual_hidden1, visual_hidden_temporal1 = self.encode_temporal(visual_hidden1, video_mask)  # shape=(B,T,D)
                visual_hidden2, visual_hidden_temporal2 = self.encode_temporal(visual_hidden2, video_mask)  # shape=(B,T,D)

                return visual_hidden1, visual_hidden2, visual_hidden_temporal1, visual_hidden_temporal2
            else:
                fold = visual_hidden.size(2) // select_num
                select_idx = torch.arange(start=0, end=visual_hidden.size(2), step=fold)
                visual_hidden = visual_hidden[:, :, select_idx, :]
                visual_hidden, visual_hidden_temporal = self.encode_temporal(visual_hidden, video_mask)  # shape=(B,T,D)
                return visual_hidden
              

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False,
                                   video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output

    def get_sequence_visual_output_m(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False,
                                     video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        # encode text momentum begin ######################
        bs_pair = input_ids.size(0)
        sequence_output_m = self.clip_m.encode_text(input_ids).float()
        sequence_output_m = sequence_output_m.view(bs_pair, -1, sequence_output_m.size(-1))
        # encode text momentum end ######################

        # encode video momentum begin #####################
        visual_output_m = self.clip_m.encode_image(video, video_frame=video_frame).float()  # shape=(B*T,L,D)
        visual_output_m = visual_output_m.view(bs_pair, -1, visual_output_m.size(-2),
                                               visual_output_m.size(-1))  # shape=(B,T,L,D)

        # select some spatial tokens
        select_num = 5
        if hasattr(self, 'spatial_aggregator'):
            B, T, L, D = visual_output_m.shape
            x_cls = visual_output_m[:, :, :1, :]
            x_s = self.spatial_aggregator(visual_output_m[:, :, 1:, :].view(B * T, L - 1, D))
            x_s = x_s.view(B, T, -1, D)
            visual_output_m = torch.cat([x_cls, x_s], dim=2)
        else:
            if self.training:
                # random select
                select_idx = torch.randperm(visual_output_m.size(2) - 1)[:select_num - 1] + 1
                select_idx = torch.cat([torch.zeros([1], dtype=torch.long), select_idx], dim=0)
                visual_output_m = visual_output_m[:, :, select_idx, :]
            else:
                fold = visual_output_m.size(2) // select_num
                select_idx = torch.arange(start=0, end=visual_output_m.size(2), step=fold)
                visual_output_m = visual_output_m[:, :, select_idx, :]

        # encode temporal
        visual_output_m_temporal = self.temporal_transformer_m(visual_output_m, video_mask)
        visual_output_m = visual_output_m_temporal + visual_output_m
        visual_output_m = visual_output_m[:, :, 0, :]
        # encode video momentum end #####################
        return sequence_output_m, visual_output_m

    def get_momentum_loss(self, sequence_output, visual_output, video_mask, sequence_output_m, visual_output_m):
        with torch.no_grad():
            sequence_output_all = torch.cat([sequence_output_m, self.text_queue.clone().detach()], dim=0)
            video_mask_all = torch.cat([video_mask, self.video_mask_queue.clone().detach()], dim=0)
            visual_output_all = torch.cat([visual_output_m, self.video_queue.clone().detach()], dim=0)
            sim_t2v_m = self._get_frame_text_similarity_logits(sequence_output_m, visual_output_all, video_mask_all)
            sim_v2t_m = self._get_frame_text_similarity_logits(sequence_output_all, visual_output_m, video_mask).T

            sim_targets = torch.zeros(sim_t2v_m.size()).to(sim_t2v_m.device)
            sim_targets.fill_diagonal_(1)
            sim_targets_t2v = self.alpha * F.softmax(sim_t2v_m, dim=1) + (1 - self.alpha) * sim_targets
            sim_targets_v2t = self.alpha * F.softmax(sim_v2t_m, dim=1) + (1 - self.alpha) * sim_targets

        sim_t2v = self._get_frame_text_similarity_logits(sequence_output, visual_output_all, video_mask_all)
        sim_v2t = self._get_frame_text_similarity_logits(sequence_output_all, visual_output, video_mask).T

        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * sim_targets_t2v, dim=1).mean()
        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * sim_targets_v2t, dim=1).mean()

        loss_mom = (loss_v2t + loss_t2v) / 2
        return loss_mom

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask, ):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _get_global_similarity_logits(self, sequence_output, visual_output, video_mask):
        '''
        sequence_output: shape is (B, D)
        visual_output: shape is (B, T, D)
        video_mask: shape is (B, T)
        '''
        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())

        return retrieve_logits

    def _get_frame_text_similarity_logits(self, sequence_output, visual_output, video_mask, no_scale=False):
        '''
        sequence_output: shape is (B, 1, D)
        visual_output: shape is (B, T, D)
        video_mask: shape is (B, T)
        '''
        sequence_output = sequence_output.squeeze(1)  # (B, D)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)  # (B, T, D)
        video_mask_un = video_mask.to(dtype=torch.bool).unsqueeze(-1).permute(2, 1, 0)

        similarity_matrix = torch.einsum('nk,mjk->njm', sequence_output, visual_output)  # (B, T, B)

        similarity_matrix_weight = similarity_matrix.detach() * video_mask_un  # (B, T, B)
        similarity_matrix_weight = similarity_matrix_weight / similarity_matrix_weight.norm(dim=1, keepdim=True)
        similarity_matrix_weight = similarity_matrix_weight.masked_fill_(~video_mask_un, -1e20)
        similarity_matrix_weight = torch.softmax(4 * similarity_matrix_weight, dim=1)
        similarity_matrix = similarity_matrix_weight * similarity_matrix
        similarity_matrix = torch.sum(similarity_matrix, dim=1)

        if no_scale:
            return similarity_matrix

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * similarity_matrix

        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False,
                              loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        retrieve_logits = self._get_frame_text_similarity_logits(sequence_output, visual_output, video_mask)

        return retrieve_logits

    def get_temporal_simsiam(self, visual_output_temporal1, visual_output_temporal2, video_mask):
        '''
        visual_output_temporal (B,T,D)
        video_mask (B,T)
        '''
        p1 = self.temporal_predictor(visual_output_temporal1) 
        p1 = self._mean_pooling_for_similarity_visual(p1, video_mask)  # BxD
        p2 = self.temporal_predictor(visual_output_temporal2) 
        p2 = self._mean_pooling_for_similarity_visual(p2, video_mask)  # BxD

        z1 = self._mean_pooling_for_similarity_visual(visual_output_temporal1, video_mask)
        z2 = self._mean_pooling_for_similarity_visual(visual_output_temporal2, video_mask)

        return p1, p2, z1.detach(), z2.detach()
    
    def get_mm_simsiam(self, sequence_output, visual_output, video_mask):
        p1 = self.text_predictor(sequence_output) # (B, 1, D)
        p2 = self.video_predictor(visual_output) # (B, T, D)

        z1 = sequence_output.detach()
        z2 = visual_output.detach()

        sim_t2v_simsiam = self._get_frame_text_similarity_logits(p1, z2, video_mask, True)
        sim_v2t_simsiam = self._get_frame_text_similarity_logits(z1, p2, video_mask, True).T

        return sim_t2v_simsiam, sim_v2t_simsiam

        