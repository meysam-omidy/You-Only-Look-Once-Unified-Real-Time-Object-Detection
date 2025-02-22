import numpy as np
import torch

def batch_iou(main_boxes, pred_boxes):
    if len(pred_boxes.shape) == 4:
        main_boxes = main_boxes.reshape(-1, 1, 1, 1, 4)
        pred_boxes = np.expand_dims(pred_boxes, 0)
    else:
        main_boxes = np.expand_dims(main_boxes, 1)
        pred_boxes = np.expand_dims(pred_boxes, 0)
    xx1 = np.maximum(main_boxes[..., 0], pred_boxes[..., 0])
    yy1 = np.maximum(main_boxes[..., 1], pred_boxes[..., 1])
    xx2 = np.minimum(main_boxes[..., 2], pred_boxes[..., 2])
    yy2 = np.minimum(main_boxes[..., 3], pred_boxes[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((main_boxes[..., 2] - main_boxes[..., 0]) * (main_boxes[..., 3] - main_boxes[..., 1])                                      
        + (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1]) - wh)                                              
    return(o) 

def extract_real_boxes(yolo_pred_boxes, xywh_form=True):
    s = yolo_pred_boxes.shape[0]
    indexes_tensor = torch.tensor([[i,j] for i in range(s) for j in range(s)]).reshape(s, s, 1, 2).to(yolo_pred_boxes.device)
    indexes_tensor = torch.concatenate([indexes_tensor for _ in range(2)], dim=2)
    yolo_pred_boxes[:, :, :, :2] += indexes_tensor
    yolo_pred_boxes[:, :, :, :2] /= s
    if not xywh_form:
        yolo_pred_boxes[:, :, :, 0:2] = yolo_pred_boxes[:, :, :, 0:2] - yolo_pred_boxes[:, :, :, 2:4] / 2
        yolo_pred_boxes[:, :, :, 2:4] = yolo_pred_boxes[:, :, :, 0:2] + yolo_pred_boxes[:, :, :, 2:4]
    return yolo_pred_boxes

def non_maximum_suppression(boxes, confidences, threshold):
    iou = batch_iou(boxes, boxes)
    final_boxes_indexes = []
    for i in range(len(boxes)):
        threshed_indexes = set(np.where(iou[i] > threshold)[0].tolist()).difference(set([i]))
        if len(threshed_indexes) == 0:
            final_boxes_indexes.append(i)
        else:
            threshed_boxes = boxes[list(threshed_indexes)]
            threshed_confidences = boxes[list(threshed_indexes)]
            threshed_boxes_square = (threshed_boxes[:, 2] - threshed_boxes[:, 0])  * (threshed_boxes[:, 3] - threshed_boxes[:, 1])
            if (confidences[i] > threshed_confidences).all() or ((boxes[i, 2] - boxes[i, 0])  * (boxes[i, 3] - boxes[i, 1]) > threshed_boxes_square).all():
                final_boxes_indexes.append(i)
    return np.array(final_boxes_indexes)

def postprocess(boxes, boxes_confidence, grid_probs, threshold):
    s, _, b, _ = boxes.shape
    boxes = extract_real_boxes(boxes, xywh_form=False).detach().cpu().numpy().reshape(-1, 4)
    boxes_confidence = boxes_confidence.detach().cpu().numpy().reshape(-1)
    grid_probs = grid_probs.detach().cpu().numpy().argmax(axis=-1).reshape(s, s, 1)
    classes = np.concatenate([grid_probs, grid_probs], axis=-1).reshape(-1)
    final_boxes_indexes = non_maximum_suppression(boxes, boxes_confidence, threshold)
    final_boxes_indexes = final_boxes_indexes[np.where(logical_and(
        np.all(boxes[final_boxes_indexes] > 0, axis=-1), 
        boxes[final_boxes_indexes][:, 0] < boxes[final_boxes_indexes][:, 2],
        boxes[final_boxes_indexes][:, 1] < boxes[final_boxes_indexes][:, 3]
        ))[0]]
    final_boxes = np.zeros(shape=(len(final_boxes_indexes), 6))
    for i in range(len(final_boxes_indexes)):
        xt, yt, xb, yb = boxes[final_boxes_indexes[i]]
        final_boxes[i] = [xt, yt, xb, yb, boxes_confidence[final_boxes_indexes[i]], classes[final_boxes_indexes[i]]]
    return final_boxes

def logical_and(*arrays):
    out = np.array([True for i in range(len(arrays[0]))])
    for array in arrays:
        out = np.logical_and(out, array)
    return out
