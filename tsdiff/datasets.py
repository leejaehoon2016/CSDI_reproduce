from typing import Any, Dict, Tuple, Type

import os
import tarfile
import numpy as np
from pathlib import Path
from urllib import request

from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path
from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.transform import (
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    MapTransformation,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.model.forecast import SampleForecast


default_dataset_path: Path = get_download_path() / "datasets"
wiki2k_download_link: str = "https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz"  # noqa: E501
data_info = {
    "solar_nips" : {
        "context_length": 336,
        "prediction_length": 24,
        "freq": 'H',
        'lags_seq': [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]],
    },
}


class AddMeanAndStdFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        dtype: Type = np.float32,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data[self.feature_name] = np.array(
            [data[self.target_field].mean(), data[self.target_field].std()]
        )

        return data


def create_transforms(
    num_feat_dynamic_real,
    num_feat_static_cat,
    num_feat_static_real,
    time_features,
    prediction_length,
):
    remove_field_names = []
    if num_feat_static_real == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if num_feat_dynamic_real == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
            if not num_feat_static_cat > 0
            else []
        )
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
            if not num_feat_static_real > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=int,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
            ),
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            AddMeanAndStdFeature(
                target_field=FieldName.TARGET,
                output_field="stats",
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if num_feat_dynamic_real > 0
                    else []
                ),
            ),
        ]
    )


def create_splitter(past_length: int, future_length: int, mode: str = "train"):
    if mode == "train":
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=1,
            min_past=past_length,
            min_future=future_length,
        )
    elif mode == "val":
        instance_sampler = ValidationSplitSampler(min_future=future_length)
    elif mode == "test":
        instance_sampler = TestSplitSampler()

    splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=past_length,
        future_length=future_length,
        time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
    )
    return splitter


def get_gts_dataset(dataset_name, batch_size = 16, num_batches_per_epoch = 128):
    if dataset_name == "wiki2000_nips":
        wiki_dataset_path = default_dataset_path / dataset_name
        Path(default_dataset_path).mkdir(parents=True, exist_ok=True)
        if not wiki_dataset_path.exists():
            tar_file_path = wiki_dataset_path.parent / f"{dataset_name}.tar.gz"
            request.urlretrieve(
                wiki2k_download_link,
                tar_file_path,
            )

            with tarfile.open(tar_file_path) as tar:
                tar.extractall(path=wiki_dataset_path.parent)

            os.remove(tar_file_path)
        dataset = load_datasets(
            metadata=wiki_dataset_path / "metadata",
            train=wiki_dataset_path / "train",
            test=wiki_dataset_path / "test",
        )
    else:
        dataset = get_dataset(dataset_name)
    info = data_info[dataset_name]

    traindata = dataset.train

    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=time_features_from_frequency_str(info['freq']) if info['freq'] is not None else [],
        prediction_length=info["prediction_length"],
    )
    training_splitter = create_splitter(
        past_length=info["context_length"] + max(info['lags_seq']),
        future_length=info["prediction_length"],
        mode="train",
    )
    
    training_data = dataset.train
    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    transformed_data = transformation.apply(training_data, is_train=True)
    train_val_splitter = OffsetSplitter(offset=-info["prediction_length"] * num_rolling_evals)
    _, val_gen = train_val_splitter.split(training_data)
    val_data = val_gen.generate_instances(info["prediction_length"], num_rolling_evals)
    
    trainloader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=num_batches_per_epoch,
    )
    
    for t in trainloader:
        import pdb ; pdb.set_trace()
        t = t
    
    # for t in val_data:
    #     import pdb ; pdb.set_trace()
    #     t = t

    import pdb ; pdb.set_trace()

