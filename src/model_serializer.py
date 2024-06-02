import os
import uuid

from src.keras import keras
from src.ml_model import MlModel


class ModelSerializer:

    def serialize(self, model: MlModel) -> bytes:
        tmp_file_name = uuid.uuid4().hex
        model.model.save(tmp_file_name)
        with open(tmp_file_name, "rb") as f:
            serialized_model = f.read()
        os.remove(tmp_file_name)
        return serialized_model

    def deserialize(self, serialized_model: bytes) -> MlModel:
        tmp_file_name = uuid.uuid4().hex
        with open(tmp_file_name, "wb") as f:
            f.write(serialized_model)
        model = keras.models.load_model(tmp_file_name)
        os.remove(tmp_file_name)
        return MlModel(model)
