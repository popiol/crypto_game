import os
import uuid

from tensorflow import keras


class ModelSerializer:

    def serialize(self, model: keras.Model) -> bytes:
        tmp_file_name = uuid.uuid4().hex
        model.save(tmp_file_name)
        with open(tmp_file_name, "rb") as f:
            serialized_model = f.read()
        os.remove(tmp_file_name)
        return serialized_model

    def deserialize(self, serialized_model: bytes) -> keras.Model:
        tmp_file_name = uuid.uuid4().hex
        with open(tmp_file_name, "wb") as f:
            f.write(serialized_model)
        model = keras.models.load_model(tmp_file_name)
        os.remove(tmp_file_name)
        return model
