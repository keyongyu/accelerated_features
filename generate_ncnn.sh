python3 jit_trace.py; rm -fr kkk/* ; cp xfeat.jit.pt kkk/
pnnx kkk/xfeat.jit.pt "inputshape=[1,1,736,736]" batchaxis=0
