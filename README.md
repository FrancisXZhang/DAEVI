# DAEVI (Depth-Aware Endoscopic Video Inpainting)
The code repository for Depth-Aware Endoscopic Video Inpainting. The pre-trained model for our paper could be found at [here](https://drive.google.com/drive/folders/1vYsUxgmtsQ7V61nIbGEf2kANzsp8Cj9D?usp=sharing).

![image](Image/Framework.png)

## Training
```
python train.py --model DAEVI --config {Your Config File Path}.json
```

## Inference
```
python test.py  --gpu 0 --overlaid --output results/DAEVI_Output/ --frame datasets/EndoSTTN_dataset/JPEGImages --mask datasets/EndoSTTN_dataset/Annotations --model DAEVI -c release_model/DAEVI_24g -cn 20 --zip --ref_num 10
```
## References
- Repository: [Endo-STTN](https://github.com/endomapper/Endo-STTN).

## Citing

If you find this work useful, please consider our paper to cite:

```
@inproceedings{zhang24Depth,
 author={Zhang, Francis Xiatian and Chen, Shuang and Xie, Xianghua and Shum, Hubert P. H.},
 booktitle={Proceedings of the 2024 International Conference on Medical Image Computing and Computer Assisted Intervention},
 series={MICCAI '24},
 title={Depth-Aware Endoscopic Video Inpainting},
 year={2024},
 publisher={Springer},
 location={Marrakesh, Morocco},
}
```
