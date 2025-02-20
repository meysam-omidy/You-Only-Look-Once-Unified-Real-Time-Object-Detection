from scipy.optimize import linear_sum_assignment
import torch
from utils import batch_iou, extract_real_boxes

class YOLO_Loss(torch.nn.Module):
    def __init__(self, lambda_coord, lambda_no_obj):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        
    def forward(self, main_boxes, pred_boxes, boxes_confidence, grid_main_probs, grid_pred_probs):
        s, b = pred_boxes.shape[0], pred_boxes.shape[2]
        pred_boxes = extract_real_boxes(pred_boxes)
        cost_matrix = batch_iou(main_boxes.detach().cpu().numpy(), pred_boxes.detach().cpu().numpy()).reshape(main_boxes.shape[0], -1)
        main_matched, pred_matched = linear_sum_assignment(cost_matrix)
        main_matched, pred_matched = list(main_matched), list(pred_matched)
        loss = 0
        for i in range(s):
            for j in range(s):
                for k in range(b):
                    pred_index = i * s * b + j * b + k
                    if pred_index in pred_matched:
                        main_index = main_matched[pred_matched.index(pred_index)]
                        xy_main = main_boxes[main_index][:2]
                        xy_pred = pred_boxes[i, j, k][:2]
                        loss += self.lambda_coord * ((xy_main - xy_pred) ** 2).sum()
                        wh_main = torch.nan_to_num(main_boxes[main_index][2:].sqrt(), 0)
                        wh_predicted = torch.nan_to_num(pred_boxes[i, j, k][2:].sqrt(), 0)
                        loss += self.lambda_coord * ((wh_main - wh_predicted) ** 2).sum()
                        loss += (cost_matrix[main_index, pred_index] - boxes_confidence[i, j, k]) ** 2
                    else:
                        loss += self.lambda_no_obj * boxes_confidence[i, j, k] ** 2
                loss += ((grid_main_probs[i, j] - grid_pred_probs[i, j]) ** 2).sum()
        return loss