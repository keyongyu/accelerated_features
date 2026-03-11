import cv2
import numpy as np
import torch

from modules.kyxfeat import XFeat
import torch.nn.functional as F

def get_top_k_keypoints(weighted_scores, feat, top_k, use_nms=True):
    """
    Extracts the Top-K elements using fully vectorized PyTorch operations with
    sub-pixel refinement.

    Args:
        weighted_scores (torch.Tensor): Shape (H, W)
        feat (torch.Tensor): Shape (H, W, 64) - Descriptor map
        top_k (int): Maximum number of keypoints to return
        use_nms (bool): Whether to use 3x3 local maximum suppression

    Returns:
        dict: {
            'keypoints': torch.Tensor (K, 2) -> [x, y] (floats),
            'scores': torch.Tensor (K,),
            'descriptors': torch.Tensor (K, 64)
        }
    """
    device = weighted_scores.device
    H, W = weighted_scores.shape

    # 1. Candidate Selection (NMS or all pixels)
    if use_nms:
        # Standardize score input to (1, 1, H, W) for max_pool2d
        scores_4d = weighted_scores.view(1, 1, H, W)
        max_scores_4d = F.max_pool2d(scores_4d, kernel_size=3, stride=1, padding=1)
        max_scores = max_scores_4d.view(H, W)
        mask = (weighted_scores == max_scores) & (weighted_scores>0.000001)
    else:
        # Without NMS, all pixels are candidates
        #mask = torch.ones_like(weighted_scores, dtype=torch.bool)
        mask = weighted_scores>0.000001

    # Flatten for global indexing
    scores_flat = weighted_scores.view(-1)
    mask_flat = mask.view(-1)

    # Get indices of all valid candidates
    candidate_indices = torch.nonzero(mask_flat).squeeze(1)

    if candidate_indices.numel() == 0:
        return {
            'keypoints': torch.empty((0, 2), device=device),
            'scores': torch.empty(0, device=device),
            'descriptors': torch.empty((0, 64), device=device)
        }

    candidate_scores = scores_flat[candidate_indices]

    # 2. Global Top-K Selection
    actual_k = min(top_k, candidate_scores.size(0))
    top_scores, rel_indices = torch.topk(candidate_scores, actual_k, sorted=True)

    # 3. Coordinate Calculation
    top_indices = candidate_indices[rel_indices]
    top_y = torch.div(top_indices, W, rounding_mode='floor')
    top_x = top_indices % W

    # 4. Vectorized Sub-pixel Refinement (Quadratic Fitting)
    # Avoid points on the absolute edge for neighbor access
    within_bounds = (top_x > 0) & (top_x < W - 1) & (top_y > 0) & (top_y < H - 1)

    refined_x = top_x.float().clone()
    refined_y = top_y.float().clone()

    if within_bounds.any():
        ref_y = top_y[within_bounds]
        ref_x = top_x[within_bounds]

        # Current, Left, Right, Up, Down scores
        s_mid = weighted_scores[ref_y, ref_x]
        s_left = weighted_scores[ref_y, ref_x - 1]
        s_right = weighted_scores[ref_y, ref_x + 1]
        s_up = weighted_scores[ref_y - 1, ref_x]
        s_down = weighted_scores[ref_y + 1, ref_x]

        eps = 1e-8
        # Quadratic interpolation formula
        dx = (s_right - s_left) / (2 * (2 * s_mid - s_left - s_right) + eps)
        dy = (s_down - s_up) / (2 * (2 * s_mid - s_up - s_down) + eps)

        refined_x[within_bounds] += torch.clamp(dx, -0.5, 0.5)
        refined_y[within_bounds] += torch.clamp(dy, -0.5, 0.5)

    # Stack into (K, 2) tensor [x, y]
    keypoints = torch.stack([refined_x, refined_y], dim=1)

    # 5. Vectorized Descriptor Extraction from (H, W, 64)
    # feat is indexed directly with [top_y, top_x, :]
    top_descriptors = feat[top_y, top_x, :]

    return {
        'keypoints': keypoints,
        'scores': top_scores,
        'descriptors': top_descriptors
    }


xfeat = XFeat(top_k=3000)

useNCNN=1
thr = 0.55
def mydetectAndComputeOrig(x):
    input = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()
    current = xfeat.detectAndCompute(input, top_k=5000, detection_threshold=thr)[0]
    kpts, scores, descs = current["keypoints"], current["scores"], current["descriptors"]
    print(f"scores: {scores}")
    print(f"descs: {descs}")
    return kpts, descs
    # idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
    # points1 = kpts1[idx0].cpu().numpy()
    # points2 = kpts2[idx1].cpu().numpy()

def mydetectAndComputeNCNN(x):
    input = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()
    weighted_scores, feat = xfeat.detectAndComputeNCNN(input, thr)
    current = get_top_k_keypoints(weighted_scores[0], feat[0], 5000, True)
    kpts, scores, descs = current["keypoints"], current["scores"], current["descriptors"]
    print(f"scores: {len(scores)}, data: {scores}")
    print(f"kpts: {kpts}")
    print(f"descs: {descs}")
    return kpts, descs


mydetectAndCompute = mydetectAndComputeNCNN if useNCNN else mydetectAndComputeOrig
# 1. Load image
def load_grey(image_path:str):
    #image_path = "key_0001.jpg"
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def testMatching():
    img1=load_grey("key_0001.jpg")
    img2=load_grey("key_0004.jpg")

    # 2. Detect keypoints (Example using ORB)
    # orb = cv2.ORB_create()
    # keypoints = orb.detect(img, None)
    kpts1, descs1 = mydetectAndCompute(img1)
    kpts2, descs2 = mydetectAndCompute(img2)
    idx0, idx1 = xfeat.match(descs1, descs2, 0.82)
    points1 = kpts1[idx0].cpu().numpy()
    points2 = kpts2[idx1].cpu().numpy()
    print(f"detect :kpts1:{len(kpts1)}, kpts2:{len(kpts2)}")
    print(f"xfeat.match:  pts1[0:10]:{points1[0:10]}")
    print(f"xfeat.match:  pts2[0:10]:{points2[0:10]}")
    if len(points1) > 10 and len(points2) > 10:
        # Find homography
        H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 4.0, maxIters=700, confidence=0.995)
        inliers = inliers.flatten() > 0
        if inliers.sum() < 50:
            H = None
        print(f"matching:  pts1:{len(points1)}, pts2:{len(points2)}, inliers number:{inliers.sum()}")
        #if self.args.method in ["SIFT", "ORB"]:
        #    good_matches = [m for i,m in enumerate(matches) if inliers[i]]
        kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
        kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
        good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

        # Draw matches
        matched_frame = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
        cv2.imshow("matches", matched_frame)
        cv2.waitKey(0)


def testMatchingSIFT():
    img1=load_grey("key_0001.jpg")
    img2=load_grey("key_0004.jpg")
    sift=cv2.SIFT_create(3000, contrastThreshold=-1, edgeThreshold=1000)
    matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # 2. Detect keypoints (Example using ORB)
    # orb = cv2.ORB_create()
    # keypoints = orb.detect(img, None)
    matches, good_matches = [], []
    kp1, kp2 = [], []
    points1, points2 = [], []
    kp1, descs1 = sift.detectAndCompute(img1, None)
    kp2, descs2 = sift.detectAndCompute(img2, None)
    print(f"kp1 number:{len(kp1)}, kp2 number:{len(kp2)}")

    matches = matcher.match(descs1, descs2)
    points1,points2 = [],[]
    if len(matches)>10:
        if len(matches) > 10:
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt

    if len(points1) > 10 and len(points2) > 10:
        # Find homography
        H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 4.0, maxIters=700, confidence=0.995)
        inliers = inliers.flatten() > 0
        if inliers.sum() < 50:
            H = None
        print(f"inliers number:{inliers.sum()}")
        good_matches = [m for i,m in enumerate(matches) if inliers[i]]
        # Draw matches
        matched_frame = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
        cv2.imshow("matches", matched_frame)
        cv2.waitKey(0)

def testKeyPoint():
    img=load_grey("key_0001.jpg")
    kpts, descs = mydetectAndCompute(img)
    npkpts = kpts.cpu().numpy()
    print(f"keypoints: {npkpts}")
    print(f"num of keypoints: {len(npkpts)}")
    keypoints = [cv2.KeyPoint(p[0], p[1], 5) for p in npkpts]

    # 3. Draw keypoints
    # Note: outImage can be None to return a new image
    img_with_keys = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imshow("Keypoints", img_with_keys)
    cv2.waitKey(0)

#testKeyPoint()
testMatching()
#testMatchingSIFT()
