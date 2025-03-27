# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import torch
import pytorch_lightning as pl
from lib.loss import InfoNCE
import torchmetrics
from lib.imu_transforms import *
import torch
import random

class FeatureQueue():
    def __init__(self, size, dim, device):
        self.size = size
        self.dim = dim
        self.queue = torch.zeros((size, dim), dtype=torch.float32).cuda()
        self.ptr = 0
        self.device = device
    
    def enqueue(self, tensors):
        """
        Enqueue a batch of tensors.

        Args:
            tensors (torch.Tensor): A batch of tensors of shape (batch_size, dim).
        """
        batch_size = tensors.size(0)
        if batch_size > self.size:
            raise ValueError("Batch size cannot be larger than queue size.")
        
        end_ptr = (self.ptr + batch_size) % self.size

        if end_ptr > self.ptr:
            self.queue[self.ptr:end_ptr] = tensors
        else:
            part1_len = self.size - self.ptr
            self.queue[self.ptr:] = tensors[:part1_len]
            self.queue[:end_ptr] = tensors[part1_len:]
        
        self.ptr = end_ptr
    
    def to(self, device):
        self.queue = self.queue.to(device)
        self.device = device

    def dequeue_and_enqueue(self, tensors):
        """
        Dequeue and enqueue a batch of tensors.

        Args:
            tensors (torch.Tensor): A batch of tensors of shape (batch_size, dim).

        Returns:
            torch.Tensor: The updated queue.
        """
        self.enqueue(tensors)
        return self.queue
    
    def get_queue(self):
        return self.queue
    
    def find_nearest_neighbors(self, tensors, k):
        """
        Find the k nearest neighbors in the queue for each tensor in the batch.

        Args:
            tensors (torch.Tensor): A batch of tensors of shape (batch_size, dim).
            k (int): The number of nearest neighbors to find.

        Returns:
            torch.Tensor: Indices of the k nearest neighbors for each tensor in the batch.
        """
        # Ensure tensors is on the same device as the queue
        tensors = tensors.to(self.device)

        # Compute cosine similarity for each tensor in the batch with the entire queue
        similarities = torch.nn.functional.cosine_similarity(
            tensors.unsqueeze(1),  # Shape: (batch_size, 1, dim)
            self.queue.unsqueeze(0),  # Shape: (1, size, dim)
            dim=2  # Compute similarity along the feature dimension
        )

        _, top_k_indices = torch.topk(similarities, k, dim=1)
        return top_k_indices
    
    def get_feats_at_indices(self, indices):
        """
        Get the features at the specified indices.

        Args:
            indices (torch.Tensor): Indices of the features to retrieve.

        Returns:
            torch.Tensor: Features at the specified indices.
        """
        return self.queue[indices]



# LightningModule: 모델 정의 및 훈련/테스트 절차 포함
class PRIMUSLearningModule(pl.LightningModule):
    def __init__(self, modality_to_encoder, source_modality, target_modalities, ssl_coeff, nnclr, transform_indices=[1, 4], multihead=False):
        """
        modality_to_encoder = {
                'imu': imu_encoder,
                'text': text_encoder,
                'video': video_encoder,
                'audio': audio_encoder,
            }
        """
        super().__init__()

        self.source_modality = source_modality
        self.target_modalities = target_modalities
        self.list_modalities = modality_to_encoder.keys()
        self.num_views = 2 # Contrastive Learning 과정에서 각 샘플에 대해 2개의 변형된 버전을 생성하여 학습하는 것을 의미 -> 같은 샘플의 서로 다른 뷰를 **양의 쌍(positive pair)**으로, 다른 샘플과의 조합을 **음의 쌍(negative pair)**으로 사용하여 모델이 강건한 표현을 학습
        self.ssl_coeff = ssl_coeff # self supervised learning loss를 조절하는 계수 -> SSL 손실이 전체 학습에 얼마나 영향을 미치는지 조절
        self.multihead = multihead
        self.nnclr = nnclr

        imu_transforms = [
                noise_transform_vectorized, #0
                scaling_transform_vectorized, #1 -----> 변형 1번
                negate_transform_vectorized, #2
                time_flip_transform_vectorized, #3 
                time_segment_permutation_transform_improved, #4 -----> 변형 4번
                rotation_transform_vectorized, #5
            ]

        self.transform_indices = transform_indices
        
        # self.imu_transform 자체가 combined_transform_func 함수가 됨
        self.imu_transform = generate_combined_transform_function(imu_transforms, self.transform_indices)

        # multimodal contrastive loss
        self.mmcl_loss = InfoNCE(symmetric_loss=True, learn_temperature=True)
        
        # self supervised loss
        self.ssl_loss = InfoNCE(symmetric_loss=True, learn_temperature=True)

        if "imu" in self.list_modalities:
            self.imu_encoder = modality_to_encoder["imu"]

        if "text" in self.list_modalities:
            self.text_encoder = modality_to_encoder["text"]

        if "video" in self.list_modalities:
            self.video_encoder = modality_to_encoder["video"]

        # NNCLR을 위한 특징 큐(Feature Queue)를 설정하는 부분
        # 일반적인 대조 학습(Contrastive Learning)에서는 같은 배치(batch) 내에서만 positive/negative 샘플을 생성함
        # 하지만 NNCLR에서는 최근 학습된 샘플(큐에 저장된 샘플) 중에서 가장 가까운 이웃을 positive 샘플로 사용
        # 매번 전체 데이터셋에서 최근 샘플을 찾는 것은 연산적으로 비효율적이므로, FeatureQueue를 사용하면 이전 배치의 정보를 저장하고, 필요할 때 즉시 가져올 수 있음
        # 8192 → 큐에 저장할 수 있는 최대 특징 벡터 개수 (8,192개)
        # 512 → 각 특징 벡터의 차원 (512차원 임베딩)
        if self.nnclr:
            self.vid_queue = FeatureQueue(8192, 512, device=self.video_encoder.device)
            self.text_queue = FeatureQueue(8192, 512, device=self.text_encoder.device)
            self.imu_queue = FeatureQueue(8192, 512, device=self.imu_encoder.device)


    def to(self, device):
        super(PRIMUSLearningModule, self).to(device)
        if self.nnclr:
            self.vid_queue.to(device)
            self.text_queue.to(device)
            self.imu_queue.to(device)

    def fetch_from_queue(self, batch, domain='video', imu_only=None):

        imu_key = 'view=0' if not self.multihead else 'ssl_view=0'
        if imu_only is None:
            if domain == 'video':
                top_k_indices = self.vid_queue.find_nearest_neighbors(batch['video'], 1)
            elif domain == 'text':
                top_k_indices = self.text_queue.find_nearest_neighbors(batch['text'], 1)
            elif domain == 'imu':
                top_k_indices = self.imu_queue.find_nearest_neighbors(batch[imu_key], 1)
            else:
                raise ValueError('Invalid domain')

            bsz = batch['video'].shape[0]
            vid_feats = self.vid_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)
            text_feats = self.text_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)
            imu_feats = self.imu_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)

            # Update the queue
            self.vid_queue.enqueue(batch['video'].detach())
            self.text_queue.enqueue(batch['text'].detach())
            self.imu_queue.enqueue(batch[imu_key].detach())

            return vid_feats, text_feats, imu_feats
        else:
            top_k_indices = []
            cache_batch = {'video': [], 'text': [], imu_key: []}
            for i, io in enumerate(imu_only):
                if io:
                    top_k_indices.append(self.imu_queue.find_nearest_neighbors(batch[imu_key][i].unsqueeze(0), 1))
                else:
                    if domain == 'video':
                        top_k_indices.append(self.vid_queue.find_nearest_neighbors(batch['video'][i].unsqueeze(0), 1))
                    elif domain == 'text':
                        top_k_indices.append(self.text_queue.find_nearest_neighbors(batch['text'][i].unsqueeze(0), 1))
                    else:
                        raise ValueError('Invalid domain')
                    
                    cache_batch['video'].append(batch['video'][i].detach())
                    cache_batch['text'].append(batch['text'][i].detach())
                    cache_batch[imu_key].append(batch[imu_key][i].detach())
            
            top_k_indices = torch.cat(top_k_indices, dim=0)
            bsz = batch[imu_key].shape[0]
            vid_feats = self.vid_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)
            text_feats = self.text_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)
            imu_feats = self.imu_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)

            # Update the queue
            self.vid_queue.enqueue(torch.Tensor(cache_batch['video']))
            self.text_queue.enqueue(torch.Tensor(cache_batch['text']))
            self.imu_queue.enqueue(torch.Tensor(cache_batch[imu_key]))

            return vid_feats, text_feats, imu_feats


    # 7번 
    def forward(self, batch, train_time=False):
        # x_imu: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        if len(batch["video"]) != len(batch["narration"]) or len(batch["video"]) != len(batch["imu"]):
            print("Weird!")
            min_size = min(min(len(batch["video"]), len(batch["narration"])), len(batch["imu"]))
            batch["imu"] = batch["imu"][:min_size]
            batch["video"] = batch["video"][:min_size]
            batch["audio"] = batch["audio"][:min_size]
        """

        out = {}

        if "imu" in self.list_modalities:

            if train_time:
                
                # Contrastive Learning(대조 학습)에서 self.num_views(=2) 개의 다른 버전의 데이터를 만들어 학습하는 과정
                # 하나의 IMU 데이터에서 2개의 변형된 데이터를 생성하여 contrastive loss를 계산하는 데 사용
                for i in range(self.num_views):
                    
                    # 첫 번째 view는 원본 IMU 그대로 사용
                    if i == 0:
                        x_imu = batch["imu"]
                        
                    # 두 번째 view는 self.imu_transform를 사용하여 IMU 데이터를 랜덤 변환
                    else:
                        x_imu = self.imu_transform(batch["imu"].cpu().numpy())
                        x_imu = torch.Tensor(x_imu).cuda() # INEFFICIENT!!!!

                    # IMU 인코더에 IMU 데이터를 통과시킴
                    y_imu = self.imu_encoder(x_imu)

                    # self.multihead가 True이면, 여러 개의 출력(head)을 가진 모델임
                    # "ssl": Self-Supervised Learning Loss용 출력
                    # "mmcl": Multi-Modal Contrastive Loss용 출력
                    # "emb": 최종 임베딩 벡터
                    if self.multihead:
                        out[f"ssl_view={i}"] = y_imu["ssl"]
                        out[f"mmcl_view={i}"] = y_imu["mmcl"]
                        out[f"emb_view={i}"] = y_imu["emb"]
                        
                    # self.multihead가 False이면, 전체 임베딩 y_imu를 그대로 저장
                    else:
                        out[f"view={i}"] = y_imu 

            # 검증/테스트 시에는 IMU 데이터를 변형하지 않고 원본 그대로 사용
            # 변환 없이 그대로 IMU 인코더에 입력
            else:
                x_imu = batch["imu"]
                y_imu = self.imu_encoder(x_imu)

                # 훈련 때와 동일한 과정
                if self.multihead:
                    out["ssl"] = y_imu["ssl"]
                    out["mmcl"] = y_imu["mmcl"]
                    out["emb"] = y_imu["emb"]        
                else:
                    out = y_imu        

        if "text" in self.list_modalities:
            x_narration = batch["narration"]
            y_narration = self.text_encoder.get_text_embeddings(x_narration)
            out["text"] = y_narration


        if "video" in self.list_modalities:
            x_video = batch["video"]
            y_video = self.video_encoder.get_video_embeddings(x_video)

            # Loop through batch['video_cache_name'] and save the video embeddings 
            if "video_cache_name" in batch:
                if len(batch["video_cache_name"]) > 0:
                    for i, video_cache_name in enumerate(batch["video_cache_name"]):
                        torch.save(y_video[i], video_cache_name)
                
            out["video"] = y_video

        return out

    # 4번
    # 배치 단위로 PRIMUSLearningModule.training_step()이 호출됨
    # 이후 self._shared_eval(batch, batch_idx, "train") 실행
    def training_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "train")

    # 11번
    # 배치 단위로 PRIMUSLearningModule.validation_step()이 호출됨
    # 이후 self._shared_eval(batch, batch_idx, "val") 실행
    def validation_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch, train_time=False)

        # Prepare metrics computation
        y_query_modality = y[self.source_modality]
        loss_output = 0.0
        metrics = {}

        # Compute metrics for source modality <> each target modality
        for target_modality in self.target_modalities:
            y_key_modality = y[target_modality]
            s2t_loss = self.loss(query=y_query_modality, positive_key=y_key_modality)
            loss_output += s2t_loss
            s_t_accuracy, t_s_accuracy = evaluate_batch_similarity(
                y_query_modality, y_key_modality, device=self.device
            )

            str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                source_modality_initial=self.source_modality[0],
                target_modality_initial=target_modality[0],
            )
            str_t2s = "{target_modality_initial}2{source_modality_initial}".format(
                target_modality_initial=target_modality[0],
                source_modality_initial=self.source_modality[0],
            )
            metrics[f"{str_s2t}_accuracy"] = s_t_accuracy
            metrics[f"{str_t2s}_accuracy"] = t_s_accuracy

        # Save and log metrics
        metrics["test_loss"] = loss_output
        self.log_dict(metrics, logger=True)
        return metrics

    def predict_step(self, batch, batch_idx: int):
        return self(batch)

    # 5번
    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        
        # 6번    
        # self(batch, train_time=True)를 호출하여 결국 PRIMUSLearningModule의 forward 함수 호출!!
        # y: {modality[str]: y_*} 형태의 딕셔너리
        # modality[str]은 imu, text, video 등의 키
        # y_*는 (B, size_embeddings) 형태의 텐서
        y = self(batch, train_time=True)

        # Use NCE loss
        # y_query_modality = y[self.source_modality]
        loss_output = 0.0

        # we will mask out MMCL loss terms for the IMU only samples
        mask = [not x for x in batch['imu_only']]
        mask = torch.Tensor(mask).to(self.device)

        # MMCL Loss
        if self.ssl_coeff < 1: 
            
            # self.target_modalities = ['text', 'video']
            for target_modality in self.target_modalities:
                y_key_modality = y[target_modality]

                # y['mmcl_view=0']을 쿼리로, y_key_modality를 positive sample로 사용하여 mmcl loss 계산
                if self.multihead:
                    s2t_loss = self.mmcl_loss(query=y['mmcl_view=0'], positive_key=y_key_modality, mask=mask)
                else:
                    s2t_loss = self.mmcl_loss(query=y['view=0'], positive_key=y_key_modality, mask=mask)

                # Log the loss
                str_s2t = "{source_modality_initial}2{target_modality_initial}".format(source_modality_initial=self.source_modality[0], target_modality_initial=target_modality[0])
                self.log(f"{prefix}_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
                
                # MMCL Loss 계산
                loss_output += (1-self.ssl_coeff)*s2t_loss

        # SSL Loss
        if self.ssl_coeff > 0:
            
            # self.num_views = 2
            for i in range(self.num_views-1):
                
                # y[f"ssl_view=0"]를 쿼리로, y[f"ssl_view=1"], y[f"ssl_view=2"] 의 view와 비교하여 contrastive learning 진행
                if self.multihead:
                    ssl_loss = self.ssl_loss(query=y[f"ssl_view=0"], positive_key=y[f"ssl_view={(i+1)}"])
                else:
                    ssl_loss = self.ssl_loss(query=y[f"view=0"], positive_key=y[f"view={(i+1)}"])

                self.log(f"{prefix}_ssl_loss", ssl_loss, logger=True, sync_dist=True)
                
                # SSL Loss 계산
                loss_output += (self.ssl_coeff)*ssl_loss

        # NN-CLR Loss
        if self.nnclr:
            
            if self.ssl_coeff == 1:
                domain = 'imu'
            elif 'video' not in self.list_modalities:
                domain = 'text'
            else:
                domain = 'video'

            # 지정된 도메인에 따라 비디오, 텍스트, IMU 특징(feature) 벡터를 가져옴
            vid_feats, text_feats, imu_feats = self.fetch_from_queue(y, domain=domain, imu_only=batch['imu_only'])

            # self.target_modalities = ['text', 'video']
            if self.ssl_coeff < 1: 
                for target_modality in self.target_modalities:
                    if target_modality == 'video':
                        y_key_modality = vid_feats
                    elif target_modality == 'text':
                        y_key_modality = text_feats
                    else:
                        raise ValueError('Invalid target modality')
                                        
                    # y['mmcl_view=0']을 쿼리로, y_key_modality를 positive sample로 사용하여 mmcl loss 계산
                    if self.multihead:
                        s2t_loss = self.mmcl_loss(query=y['mmcl_view=0'], positive_key=y_key_modality)
                    else:
                        s2t_loss = self.mmcl_loss(query=y['view=0'], positive_key=y_key_modality)

                    # Log the loss
                    str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                        source_modality_initial=self.source_modality[0],
                        target_modality_initial=target_modality[0],
                    )
                    self.log(f"{prefix}_nn_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
                    
                    # MMCL Loss 계산
                    loss_output += (1-self.ssl_coeff)*s2t_loss

            # SSL Loss
            if self.ssl_coeff > 0:
                
                # self.num_views = 2
                for i in range(self.num_views-1):
                    
                    # y[f"ssl_view=0"]를 쿼리로, y[f"ssl_view=1"], y[f"ssl_view=2"] 의 view와 비교하여 contrastive learning 진행
                    if self.multihead:
                        ssl_loss = self.ssl_loss(query=y[f"ssl_view=0"], positive_key=imu_feats)
                    else:
                        ssl_loss = self.ssl_loss(query=y[f"view=0"], positive_key=imu_feats)
                    
                    self.log(f"{prefix}_nn_ssl_loss", ssl_loss, logger=True, sync_dist=True)
                    
                    # SSL Loss 계산
                    loss_output += (self.ssl_coeff)*ssl_loss 

    
        self.log(f"{prefix}_loss", loss_output, logger=True, sync_dist=True)
        
        # 8번
        # 최종 loss 반환
        # 이후 configure_optimizers()에서 Adam 옵티마이저가 설정됨
        return loss_output

    # 9번
    # Adam 옵티마이저가 설정되고, 이후 loss.backward() ~ optimizer.zero_grad() 까지 내부적으로 호출됨
    # 이후 한 epoch이 끝나면 자동으로 EgoExo4dDataModule.val_dataloader() 호출됨
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)



class MultimodalContrastiveLearningModule(pl.LightningModule):
    def __init__(self, modality_to_encoder, source_modality, target_modalities):
        """
        modality_to_encoder = {
                'imu': imu_encoder,
                'text': text_encoder,
                'video': video_encoder,
                'audio': audio_encoder,
            }
        """
        super().__init__()

        self.source_modality = source_modality
        self.target_modalities = target_modalities
        self.list_modalities = modality_to_encoder.keys()

        self.loss = InfoNCE(symmetric_loss=True, learn_temperature=True)

        if "imu" in self.list_modalities:
            self.imu_encoder = modality_to_encoder["imu"]

        if "text" in self.list_modalities:
            self.text_encoder = modality_to_encoder["text"]

        if "video" in self.list_modalities:
            self.video_encoder = modality_to_encoder["video"]

        if "audio" in self.list_modalities:
            self.audio_encoder = modality_to_encoder["audio"]

    def forward(self, batch):
        # x_imu: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        if len(batch["video"]) != len(batch["narration"]) or len(batch["video"]) != len(batch["imu"]):
            print("Weird!")
            min_size = min(min(len(batch["video"]), len(batch["narration"])), len(batch["imu"]))
            batch["imu"] = batch["imu"][:min_size]
            batch["video"] = batch["video"][:min_size]
            batch["audio"] = batch["audio"][:min_size]
        """

        out = {}

        out['imu_only'] = batch['imu_only']

        if "imu" in self.list_modalities:
            x_imu = batch["imu"]
            y_imu = self.imu_encoder(x_imu)
            out["imu"] = y_imu

        if "text" in self.list_modalities:
            x_narration = batch["narration"]

            y_narration = self.text_encoder.get_text_embeddings(x_narration)
            out["text"] = y_narration

        if "video" in self.list_modalities:
            x_video = batch["video"]
            y_video = self.video_encoder.get_video_embeddings(x_video)

            # Loop through batch['video_cache_name'] and save the video embeddings 
            if "video_cache_name" in batch:
                if len(batch["video_cache_name"]) > 0:
                    for i, video_cache_name in enumerate(batch["video_cache_name"]):
                        torch.save(y_video[i], video_cache_name)
                
                print("Saved video embeddings")

            out["video"] = y_video

        return out

    def training_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch)

        # Prepare metrics computation
        y_query_modality = y[self.source_modality]
        loss_output = 0.0
        metrics = {}

        # Compute metrics for source modality <> each target modality
        for target_modality in self.target_modalities:
            y_key_modality = y[target_modality]
            s2t_loss = self.loss(query=y_query_modality, positive_key=y_key_modality)
            loss_output += s2t_loss
            s_t_accuracy, t_s_accuracy = evaluate_batch_similarity(
                y_query_modality, y_key_modality, device=self.device
            )

            str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                source_modality_initial=self.source_modality[0],
                target_modality_initial=target_modality[0],
            )
            str_t2s = "{target_modality_initial}2{source_modality_initial}".format(
                target_modality_initial=target_modality[0],
                source_modality_initial=self.source_modality[0],
            )
            metrics[f"{str_s2t}_accuracy"] = s_t_accuracy
            metrics[f"{str_t2s}_accuracy"] = t_s_accuracy

        # Save and log metrics
        metrics["test_loss"] = loss_output
        self.log_dict(metrics, logger=True)
        return metrics

    def predict_step(self, batch, batch_idx: int):
        return self(batch)

    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        # y: {modality[str]: y_*} where y_*: B x size_embeddings
        y = self(batch)

        # Use NCE loss
        y_query_modality = y[self.source_modality]
        loss_output = 0.0

        # Compute loss for source modality <> each target modality
        for target_modality in self.target_modalities:
            y_key_modality = y[target_modality]
            s2t_loss = self.loss(query=y_query_modality, positive_key=y_key_modality)

            # Log the loss
            str_s2t = "{source_modality_initial}2{target_modality_initial}".format(
                source_modality_initial=self.source_modality[0],
                target_modality_initial=target_modality[0],
            )
            self.log(f"{prefix}_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
            loss_output += s2t_loss

        self.log(f"{prefix}_loss", loss_output, logger=True, sync_dist=True)
        return loss_output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)


def evaluate_batch_similarity(source_embeddings, target_embeddings, device):
    """
    Given a batch matrix (size B) of paired embeddings,
    measure the accuracy of the predictions by checking nearest the neighbors
    """
    #  Compute similarity
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)

    # similarities: B x B
    similarities = torch.mm(s, t.transpose(0, 1))

    # pred: 1 x B (ideally [0, 1, 2, 3, ..., B])
    s_t_pred = torch.argmax(similarities, dim=1)
    t_s_pred = torch.argmax(similarities, dim=0)
    B = len(s_t_pred)
    s_t_accuracy = sum(s_t_pred == torch.arange(B, device=device)) / B
    t_s_accuracy = sum(t_s_pred == torch.arange(B, device=device)) / B
    return s_t_accuracy, t_s_accuracy


class ClassificationModule(pl.LightningModule):
    def __init__(self, model):
        """
        Encoder + Head
        """
        super().__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = model
        self.accuracy_train = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.accuracy_valid = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.accuracy_test = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.f1_test = torchmetrics.F1Score(task="multiclass", num_classes=4, average="macro")

    def forward(self, batch):
        # x_imu: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        in: batch_size x 6 x window_size
        out: batch_size x 1
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "train")

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log("train_acc_epoch", self.accuracy_train)

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log("val_acc_epoch", self.accuracy_valid)

    def test_epoch_end(self, outs):
        # log epoch metric
        self.log("test_acc_epoch", self.accuracy_test)
        self.log("test_f1_epoch", self.f1_test)

    def validation_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        return self._shared_eval(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int):
        return self(batch)

    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        x, y = batch
        y_hat = self(x)
        loss_output = self.loss_fn(y_hat, y)
        if prefix == "train":
            self.accuracy_train(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_train, logger=True)
        if prefix == "val":
            self.accuracy_valid(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_valid, logger=True)
        if prefix == "test":
            self.accuracy_test(y_hat, y)
            self.f1_test(y_hat, y)
            self.log(f"{prefix}_acc_step", self.accuracy_test, logger=True)
            self.log(f"{prefix}_f1_step", self.f1_test, logger=True)
        self.log(f"{prefix}_loss", loss_output, logger=True)
        return loss_output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)
