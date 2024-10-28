# PET
This is the official implementation of PET **(Personalized View Weighting with Data Enhancement Two-Pronged Contrast)** 
(CIKM 2024 Short Paper Track) 

---
### Supplementary Document
For additional data anlysis results and details about loss function, you can check here.
[Supplementary Document]
[_CIKM24__Bundle_Recommendation__Online_Appendix_.pdf](https://github.com/user-attachments/files/17536033/_CIKM24__Bundle_Recommendation__Online_Appendix_.pdf)

)


### Datasets
We use three widely used datasets for bundle recommendation, **iFashion**, **NetEase** and **Youshu**.
For the iFashion dataset, please unzip data.zip in the same folder.


---
### Run PET
```bash
cd PET
```
* **iFashion**
```bash
python train.py -d iFashion -g [gpu_id]
```
* **NetEase**
```bash
python train.py -d NetEase -g [gpu_id]
```
* **Youshu**
```bash
python train.py -d Youshu -g [gpu_id]   
```
---

### Citation
We appreciate your interest in our work. If our research contributes to your projects, please consider citing our paper:

```bibtex
@inproceedings{kim2024towards,
  title={Towards Better Utilization of Multiple Views for Bundle Recommendation},
  author={Kim, Kyungho and Kim, Sunwoo and Lee, Geon and Shin, Kijung},
  booktitle={CIKM},
  year={2024}
}
```
---
### Acknowledgement
This code is implemented based on the open source code from the paper **CrossCBR : Cross-view Contrastive Learning for Bundle Recommendation** (KDD '22).
