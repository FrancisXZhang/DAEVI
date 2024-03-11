# DAEVI
The code repository for Depth-Aware Endoscopic Video Inpainting. Currently, only the inference code is available. The full code will be released upon acceptance.

## Inference
python test.py  --gpu 0 --overlaid --output results/DAEVI_Output/ --frame datasets/EndoSTTN_dataset/JPEGImages --mask datasets/EndoSTTN_dataset/Annotations --model DAEVI -c release_model/DAEVI_24g -cn 20 --zip --ref_num 10

## References
- Repository: [Endo-STTN](https://github.com/endomapper/Endo-STTN).
