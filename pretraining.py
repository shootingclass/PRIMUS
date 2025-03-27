# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
import random
import os
from datetime import datetime
import torch
import pytorch_lightning as pl
from dataset.egoexo4d.dataloader import filter_narration, clean_narration_text
from lib.imu_models import MW2StackRNNPoolingMultihead, MW2StackRNNPooling
from lib.clip4clip_model import Clip4CLIPModel
from lib.train_modules import PRIMUSLearningModule
from lib.data_modules import EgoExo4dDataModule
from argparse import ArgumentParser
import yaml

def train(configs):

    random.seed(1234)

    # Load Model Parameters
    model_hparams = configs.get("model_hparams", {})
    model_name = model_hparams.get("model_name")
    model_suffix = model_hparams.get("model_suffix", "")
    imu_encoder_name = model_hparams.get("imu_encoder_name")
    video_encoder_name = model_hparams.get("video_encoder_name")
    window_sec = model_hparams.get("window_sec")
    target_fps = model_hparams.get("target_fps")
    datasetname = model_hparams.get("datasetname", "ego4d")
    imu_sampling_rate = model_hparams.get(
        "imu_sampling_rate", 50 if datasetname in ["ego4d", "egoexo4d"] else 1000
    )
    final_embedding_size = model_hparams.get("final_embedding_size", 512)

    # Params for the trainer
    train_hparams = configs.get("train_hparams", {})
    source_modality = train_hparams.get("source_modality")
    target_modalities = train_hparams.get("target_modalities")
    limit_train_batches = train_hparams.get("limit_train_batches")
    batch_size = train_hparams.get("batch_size")
    max_epochs = train_hparams.get("max_epochs")
    num_workers_for_dm = train_hparams.get("num_workers_for_dm")
    test_only = train_hparams.get("test_only")
    trainer_strategy = train_hparams.get("trainer_strategy")
    freeze_modalities = train_hparams.get("freeze_modalities")
    path_load_pretrained_imu_encoder = train_hparams.get(
        "path_load_pretrained_imu_encoder"
    )


    # Paths, etc.
    # path_root_save_dir = f"./saved_slip_clip4clip_nnclr_tuning/{model_name}"
    path_root_save_dir = os.path.join(args.path_root_save_dir, model_name)
    if not os.path.exists(path_root_save_dir):
        os.makedirs(path_root_save_dir)
    target_modalities.sort()
    list_modalities = [source_modality] + target_modalities
    source_modality_initial = source_modality[0]
    target_modality_initials = "".join([m[0] for m in target_modalities])
    if source_modality == "imu":
        source_encoder_name = imu_encoder_name
    if source_modality == "video":
        source_encoder_name = video_encoder_name

    suf = ""
    if args.ssl_coeff == 1:
        suf="_ssl_only"
    else:
        suf = f"_ssl_coeff={args.ssl_coeff}"

    model_identifier = (
        f"{model_name}_s_{source_modality_initial}_t_{target_modality_initials}"
        + f"_se_{source_encoder_name}_w_{window_sec}_{datasetname}_transforms={args.transform_list}{suf}"
    )


    if model_suffix != "":
        model_identifier += "_" + model_suffix
    else:
        model_identifier += "_%d" % (int(datetime.now().timestamp() % 10000))

    print("Model identifier: ", model_identifier)
    path_save_checkpoint = f"{path_root_save_dir}/{model_identifier}_best.ckpt"
    path_save_src_encoder = f"{path_root_save_dir}/{model_identifier}_src_encoder.pt"
    result_path = f"./results/{model_identifier}"
    configs["path_save_checkpoint"] = path_save_checkpoint

    if datasetname == "egoexo4d":
        print("Initializing EgoExo4dDataModule")
        
        # Initialize the data module
        dataset_params = {
            "window_sec": window_sec,
            "target_fps": target_fps,
            "list_modalities": list_modalities,
            "clean_narration_func": clean_narration_text,
            "filter_narration_func": filter_narration,
            "imu_sampling_rate": imu_sampling_rate,
        }

        datamodule = EgoExo4dDataModule(
            batch_size=batch_size,
            num_workers=num_workers_for_dm,
            pin_memory=True,
            drop_last=False,
            dataset_params=dataset_params
        )

    else:
        raise ValueError("Unknown dataset name")

    # Initialize encoder models
    text_encoder, video_encoder, imu_encoder = None, None, None
    modality_to_encoder = {}


    # 아래를 보면 텍스트 인코더와 비디오 인코더가 모두 Clip4CLIPModel(freeze=True)으로 동일함
    # 근데 어차피 Clip4CLIPModel 클래스는 내부에서 텍스트와 비디오를 각각 처리하도록 구현되어 있으므로, 신경쓸 필요 없음

    # 텍스트 인코더
    if "text" in list_modalities:
        # For now we only use a CLIP-based text model
        text_encoder = Clip4CLIPModel(freeze=True)
        modality_to_encoder["text"] = text_encoder

    # 비디오 인코더
    if "video" in list_modalities:
        # For now we only use a CLIP-based image model as a video encoder
        video_encoder = (
            Clip4CLIPModel(freeze=True) if text_encoder is None else text_encoder
        )
        video_encoder.video_encoder_name = video_encoder_name

        modality_to_encoder["video"] = video_encoder

    # IMU 인코더
    if "imu" in list_modalities:
        if args.multihead:
            imu_encoder = MW2StackRNNPoolingMultihead(size_embeddings=final_embedding_size)
        
        else:
            imu_encoder = MW2StackRNNPooling(size_embeddings=final_embedding_size)

        if path_load_pretrained_imu_encoder:
            # Load the parameters
            imu_encoder.load_state_dict(torch.load(path_load_pretrained_imu_encoder))
            print("loaded pretrained imu model")

        modality_to_encoder["imu"] = imu_encoder

    for modality in list_modalities:
        if modality in freeze_modalities:
            modality_to_encoder[modality].eval()
            print("Freezing modality: ", modality)
            modality_to_encoder[modality].freeze()

    # Initialize the training module for contrastive training
    model = PRIMUSLearningModule(
        modality_to_encoder=modality_to_encoder,
        source_modality=source_modality,
        target_modalities=target_modalities,
        ssl_coeff=args.ssl_coeff,
        multihead = args.multihead, 
        nnclr = args.nnclr
    )
    print("Initialized model...")

    # Checkpoint settings
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=path_root_save_dir,
        filename=f"{model_identifier}" + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )


    print(trainer_strategy)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        strategy=trainer_strategy,
        limit_train_batches=limit_train_batches,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0
    )
    print("Initialized trainer...")


    # 1번
    # trainer.fit()을 호출하면 자동으로 학습 루프 실행
    # 다음 시퀀스로는 EgoExo4dDataModule.setup("fit")이 호출됨!!
    print("Start training: [%s] ..." % path_save_checkpoint)
    trainer.fit(model, datamodule=datamodule)
    torch.distributed.barrier()

    # Save the checkpoint & encoder to a temp folder
    print("Best checkpoint:", checkpoint_callback.best_model_path)
    model.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        modality_to_encoder=modality_to_encoder,
        source_modality=source_modality,
        target_modalities=target_modalities,
        ssl_coeff=args.ssl_coeff,
    )
    src_encoder = None
    if source_modality == "imu":
        src_encoder = model.imu_encoder
    elif source_modality == "audio":
        src_encoder = model.audio_encoder
    elif source_modality == "video":
        src_encoder = model.video_encoder
    torch.save(src_encoder.state_dict(), path_save_src_encoder)


if __name__ == "__main__":

    # YAML 파일이 configs이자 기본 설정값이고
    # 명령줄 인자 args를 통해 configs를 업데이트하여, 최종 학습 설정을 반영하는 과정

    # ArgumentParser 객체 생성, 명령줄 인자 처리할 준비
    parser = ArgumentParser()

    # Main parameters are defined in a YAML file
    # YAML 파일의 경로를 기본 인자로 받음
    parser.add_argument(
        "--path_configs", default="./configs/train_contrastive/default.yaml"
    )

    # Override-params for a quick resource allocation adjustment or for debugging purposes
    # If it is *not* None, the values in args override the values in the YAML file.
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num_workers_for_dm", default=None)
    parser.add_argument("--max_epochs", default=None)
    parser.add_argument("--test_only", default=None)
    parser.add_argument("--path_load_pretrained_imu_encoder", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--multihead", default=False, action="store_true")
    parser.add_argument("--path_root_save_dir", default="./saved/")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--nnclr", default=False, action="store_true", help="Whether to nearest-neighbor supervision")
    parser.add_argument("--ssl_coeff", default=0.5, type=float, help="SSL loss coefficient, final loss = ssl_coeff*ssl_loss + (1-ssl_coeff)*mmcl_loss")
    parser.add_argument("--transform_list", nargs="+", help="Indices of transforms to apply to the IMU frames. Transform list: [noise_transform_vectorized, scaling_transform_vectorized, negate_transform_vectorized, time_flip_transform_vectorized, time_segment_permutation_transform_improved, rotation_transform_vectorized]")
    
    # 위에서 정의한 모든 명령줄 인자들을 받아 args 변수에 저장
    args = parser.parse_args()
    print('args', args)
    
    # Load the YAML file
    # args.path_configs에 지정된 YAML 파일을 읽어서 configs 딕셔너리에 저장
    with open(args.path_configs) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Override the configs with args, if requested
    # 명령줄 인자를 통해 YAML 설정값 덮어쓰기
    if args.gpus is not None:
        configs["train_hparams"]["gpus"] = int(args.gpus)
    if args.num_workers_for_dm is not None:
        configs["train_hparams"]["num_workers_for_dm"] = int(args.num_workers_for_dm)
    if args.max_epochs is not None:
        configs["train_hparams"]["max_epochs"] = int(args.max_epochs)
    if args.test_only is not None:
        configs["train_hparams"]["test_only"] = eval(args.test_only)
    if args.path_load_pretrained_imu_encoder is not None:
        configs["train_hparams"][
            "path_load_pretrained_imu_encoder"
        ] = args.path_load_pretrained_imu_encoder
    if args.dataset is not None:
        configs["model_hparams"]["datasetname"] = args.dataset
    if args.model_name is not None:
        configs["model_hparams"]["model_name"] = args.model_name

    print('configs', configs)

    # train(configs)