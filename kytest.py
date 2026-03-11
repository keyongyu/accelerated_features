import os
import cv2
import torch
from modules.kyxfeat import XFeat
import torch.nn.functional as F


print("torch:" + torch.__version__)

def get_top_k_keypoints(weighted_scores, feat, top_k,  use_nms=True):
    """
    Extracts the Top-K elements from a single-channel heatmap using PyTorch.

    Args:
        weighted_scores (torch.Tensor): Shape (H, W) or (1, H, W)
        feat (torch.Tensor): Shape (1, H, W, 64) - Descriptor map
        top_k (int): Maximum number of keypoints to return
        threshold (float): Minimum score threshold
        use_nms (bool): Whether to use 3x3 local maximum suppression

    Returns:
        list: List of dictionaries containing 'pt', 'score', and 'descriptor'
    """
    # Ensure scores are 2D (1, 1, H, W) for max_pool2d if needed
    if weighted_scores.ndim == 2:
        scores = weighted_scores.unsqueeze(0).unsqueeze(0)
    elif weighted_scores.ndim == 3:
        scores = weighted_scores.unsqueeze(0) # Assumes (1, H, W)
    else:
        scores = weighted_scores

    device = scores.device
    _, _, H, W = scores.shape

    feat = feat.squeeze(0)

    # 1. Thresholding
    mask = scores > 0

    # 2. Non-Maximum Suppression (3x3 local max)
    if use_nms:
        # F.max_pool2d is an extremely fast way to find local maximums in Torch
        max_scores = F.max_pool2d(scores, kernel_size=3, stride=1, padding=1)
        is_max = (scores == max_scores)
        mask = mask & is_max

    # Flatten for selection
    scores_flat = scores.view(-1)
    mask_flat = mask.view(-1)

    # Filter scores and indices by the mask
    # We use nonzero() to get indices of candidates passing threshold and NMS
    candidate_indices = torch.nonzero(mask_flat).squeeze()

    if candidate_indices.numel() == 0:
        return []

    candidate_scores = scores_flat[candidate_indices]

    # 3. Global Top-K Selection
    actual_k = min(top_k, candidate_scores.size(0))
    top_scores, rel_indices = torch.topk(candidate_scores, actual_k, sorted=True)

    # Map back to original flattened indices
    top_indices = candidate_indices[rel_indices]

    # Calculate coordinates
    # Note: index = y * W + x
    top_y = torch.div(top_indices, W, rounding_mode='floor')
    top_x = top_indices % W

    # 4. Construct Results and Attach Descriptors
    results = []

    # Move to CPU for list construction if on GPU
    top_scores_cpu = top_scores.detach().cpu().tolist()
    top_y_cpu = top_y.detach().cpu().tolist()
    top_x_cpu = top_x.detach().cpu().tolist()

    for i in range(actual_k):
        y, x = top_y_cpu[i], top_x_cpu[i]

        # Slicing the (H, W, 64) descriptor map
        # If feat is on GPU, this stays on GPU until converted to list/numpy
        descriptor = feat[y, x, :].detach().cpu().numpy()

        results.append({
            'pt': (x,y),
            'score': top_scores_cpu[i],
            'descriptor': descriptor
        })

    return results

model = XFeat()
for param in model.parameters():
    param.requires_grad = False
model.eval()
# dummy_input = torch.randn(1, 1, 1024, 768)
image_path = "key_0001.jpg"
dummy_input = cv2.imread(image_path, cv2.IMREAD_COLOR)
dummy_input = torch.from_numpy(cv2.cvtColor(dummy_input, cv2.COLOR_BGR2GRAY))
dummy_input = dummy_input.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

weighted_scores, feat = model.detectAndComputeNCNN(dummy_input, 0.95)

# print("weighted_scores shape:" + str(weighted_scores.shape))
# t = 0.4
# print(f"count>{t}:", torch.count_nonzero(weighted_scores > t).item())
# indices = torch.nonzero(weighted_scores > t)
# print(f"Indices of weighted_scores > {t}:", indices)

# mask = weighted_scores > t
# print(f"feat shape: {feat.shape}")
# print(f"mask shape: {mask.shape}")
# print("scores:")
# print(weighted_scores[mask])
# print("descriptors: ")
# print(feat[mask])




lst = get_top_k_keypoints(weighted_scores, feat, 100)

print(f"score[0,987,251]={weighted_scores[0,987,251]}")
print(f"score[0,626,603]={weighted_scores[0,626,603]}")
print(f"score[0,936,202]={weighted_scores[0,936,202]}")
print(f"{len(lst)} point found")
print(f"they are: {lst}")
