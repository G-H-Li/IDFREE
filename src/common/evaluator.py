import os

import torch
from logging import getLogger

from src.common.trainer import AbstractTrainer
from src.utils.topk_evaluator import TopKEvaluator


class Evaluator(AbstractTrainer):
    def __init__(self, config, model):
        super(Evaluator, self).__init__(config, model)
        self.logger = getLogger()
        self.device = config['device']
        self.evaluator = TopKEvaluator(config)
        self.model.to(self.device)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=True, map_location=self.device)
            if self.config['model'] == "TEST_MODEL":
                del checkpoint["v_feat_i.weight"]
                del checkpoint["t_feat_i.weight"]
            elif self.config['model'] == "SOIL":
                del checkpoint["user_embedding.weight"]
                del checkpoint["item_id_embedding.weight"]
                del checkpoint["image_embedding.weight"]
                del checkpoint["text_embedding.weight"]
            self.model.load_state_dict(checkpoint, strict=False)
            self.logger.info(f'Model loaded from {model_path}')
        else:
            self.logger.error(f'Model path {model_path} does not exist!')

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)
