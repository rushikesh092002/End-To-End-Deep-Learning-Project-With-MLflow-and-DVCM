import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import dagshub
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        print("🔍 Loading model...")
        self.model = self.load_model(self.config.path_of_model)

        print("📦 Preparing validation data generator...")
        self._valid_generator()

        print("✅ Evaluating model...")
        self.score = self.model.evaluate(self.valid_generator)
        print(f"📊 Evaluation Score: Loss = {self.score[0]}, Accuracy = {self.score[1]}")

        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
        print("💾 Scores saved to scores.json")

    def log_into_mlflow(self):
        # ✅ Initialize DagsHub logging
        dagshub.init(
            repo_owner='rushikesh092002',
            repo_name='End-To-End-Deep-Learning-Project-With-MLflow-and-DVCM',
            mlflow=True
        )

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            print("🚀 Logging parameters and metrics to MLflow (via DagsHub)...")
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })

            print("📦 Logging model to MLflow...")
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")

            print("✅ MLflow logging complete!")