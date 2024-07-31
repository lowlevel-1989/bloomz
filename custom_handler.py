# REF: https://kserve.github.io/website/0.8/modelserving/inference_api/#inference-request-examples

import os
import json
import zipfile
import logging

import torch

from abc import ABC

from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler, ABC):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        logger.info(context)

        self.manifest = context.manifest

        properties = context.system_properties

        self.device = torch.device(
            "cuda:{}".format(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        logger.info("device: {}".format(self.device))

        logger.info(properties)
        model_dir = properties.get("model_dir")

        """
        logger.info("snapshot")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                logger.info(f" - {file}")
        """

        model_metadata = None
        with open(f"{model_dir}/setup_config.json") as f:
            model_metadata = json.load(f)

        llm_pretrained_zip = os.path.join(model_dir, model_metadata["model_pretrained"])
        llm_snapshot_zip   = os.path.join(model_dir, model_metadata["model_snapshot"])

        with zipfile.ZipFile(llm_pretrained_zip) as zipf:
            zipf.extractall(f"{model_dir}/model")

        with zipfile.ZipFile(llm_snapshot_zip) as zipf:
            zipf.extractall(f"{model_dir}/model")

        model_path = "{model_dir}/model/{revision}".format(**{
            "model_dir": model_dir,
            "revision":  model_metadata["model_revision"],
        })

        self.text_column = model_metadata["text_column"]

        model  = AutoModelForCausalLM.from_pretrained(model_path)

        self.model  = PeftModel.from_pretrained(model, f"{model_dir}/model/snapshot")

        self.model.to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer

        logger.info(f"Transformer model from path {model_dir} loaded successfully")

        logger.info(self.manifest)
        logger.info(self.manifest["model"])

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """

        input_ids_batch = None
        attention_mask_batch = None

        logger.info("req: {}".format(requests))

        # TODO: soportar todas las entradas en un solo request
        # Solo me quedo con el ultimo
        for idx, data in enumerate(requests):

            # detectar torchserve
            input_text = data.get("data")

            if input_text is None:
                input_text = data.get("body")

                # detectar kserve
                if isinstance(input_text, dict):
                    input_text = input_text["inputs"][0]["data"][0]

            if isinstance(input_text, list):
                input_text = input_text[0]

            inputs = self.tokenizer(
                f"{self.text_column} : {input_text} Label : ",
                return_tensors="pt",
            )

        return inputs

    def inference(self, inputs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """

        inferences = []

        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
            )
            outputs = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

        inferences.append(outputs[0])

        logger.info("Generated text: '{}'".format(inferences))

        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """

        output = {
            "name":     "content",
            "shape":    [1],
            "datatype": "BYTES",
            "data":     [inference_output[0]]
        }

        return [{
            "id":      "bloomz",
            "outputs": [output],
        }]
