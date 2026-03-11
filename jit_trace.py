import torch
import cv2
import os
from modules.kyxfeat import XFeat
print("torch:"+torch.__version__)

model = XFeat()
for param in model.parameters():
    param.requires_grad = False
model.eval()
#dummy_input = torch.randn(1, 1, 1024, 768)
image_path = 'key_0001.jpg'
target_w=736
target_h=736
dummy_input = cv2.imread(image_path, cv2.IMREAD_COLOR)
dummy_input = cv2.resize(dummy_input,  (target_w, target_h))
dummy_input = torch.from_numpy(cv2.cvtColor(dummy_input, cv2.COLOR_BGR2GRAY))
dummy_input = dummy_input.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

# XFeat returns a dict like: {'keypoints': T, 'scores': T, 'descriptors': T}
# We wrap it to ensure ONLY the tensors are returned in a tuple.
def trace_wrapper(x):
    #out_dict = model.detectAndCompute(x, top_k=4096)
    #return out_dict[0]['keypoints'], out_dict[0]['scores'], out_dict[0]['descriptors']
    weighted_scores, feat = model.detectAndComputeNCNN(x, 0.05)
    return weighted_scores, feat
    #return out_dict[0]['keypoints'], out_dict[0]['scores'], out_dict[0]['descriptors']

    # Return as a tuple of Tensors (trace-compatible)
    # Adjust these keys based on the exact version of XFeat you are using
    #out_dict = model.detectAndComputeDense(x, top_k=4096)
    #return out_dict['keypoints'],out_dict['descriptors'], out_dict['scales']


# Trace the wrapper function
with torch.no_grad():
    traced_model = torch.jit.trace(trace_wrapper, dummy_input)

# 2. IMPORTANT: Manually set the input name to 'in0'
# PNNX looks for standard naming to anchor the batch axis
graph_inputs = list(traced_model.graph.inputs())
# graph_inputs[0].setDebugName("in0")
traced_model.save("xfeat.jit.pt")
