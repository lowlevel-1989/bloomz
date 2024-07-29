import os
import logging

logger = logging.getLogger(__name__)


class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        logger.info("Entrada Vinicio")

        logger.info(context)
        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        logger.info(properties)
        model_dir = properties.get("model_dir")

        logger.info("snapshot")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                logger.info(f" - {file}")


        logger.info(self.manifest)
        logger.info(self.manifest["model"])

        self.initialized = False


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model.forward(data)
        return pred_out
