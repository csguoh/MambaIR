<p align="center">
    <img src="assets/logo.png" width="340">
</p>

## MambaIR: A Simple Baseline for Image Restoration with State-Space Model

[[Paper](https://arxiv.org/pdf/2312.08881.pdf)]  [[Suppl]()] [Project Page][Zhihu(知乎)]


[Hang Guo](https://github.com/csguoh), Jinmin Li, [Tao Dai](https://cstaodai.com/), Zhihao Ouyang, Xudong Ren, [Shu-Tao Xia](https://scholar.google.com/citations?hl=zh-CN&user=koAXTXgAAAAJ)


> **Abstract:**  Recent years have witnessed great progress in image restoration thanks to the advancements in modern deep neural networks \textit{e.g.} Convolutional Neural Network and Transformer. However, existing restoration backbones are usually limited due to the inherent local reductive bias or quadratic computational complexity. Recently, Selective Structured State Space Model \textit{e.g.}, Mamba, have shown great potential for long-range dependencies modeling with linear complexity, but it is still under-explored in low-level computer vision. In this work, we introduce a simple but strong benchmark model, named MambaIR, for image restoration. In detail, we propose the Residual State Space Block as the core component, which employs convolution and channel attention to enhance capabilities of the vanilla Mamba. In this way, our MambaIR takes advantages of local patch recurrence prior as well as channel interaction to produce restoration-specific feature representation. Extensive experiments demonstrate the superiority of our method, for example, MambaIR outperforms Transformer-based baseline SwinIR by up to 0.36dB, using similar computational cost but with global receptive field. 


<p align="center">
    <img src="assets/pipeline.png" style="border-radius: 15px">
</p>

⭐If this work is helpful for you, please help star this repo. Thanks!🤗



## 📑 Contents

- [Visual Results](#visual_results)
- [News](#news)
- [TODO](#todo)
- [Results](#results)
- [Citation](#cite)


## <a name="visual_results"></a>:eyes:Visual Results On Different Restoration Tasks
[<img src="assets/imgsli1.png" height="153"/>](https://imgsli.com/MjI1Njk3) [<img src="assets/imgsli7.png" height="153"/>](https://imgsli.com/MjI1NzIx) [<img src="assets/imgsli5.png" height="153"/>](https://imgsli.com/MjI1NzEx) [<img src="assets/imgsli2.png" height="153"/>](https://imgsli.com/MjI1NzAw)

[<img src="assets/imgsli4.png" height="150"/>](https://imgsli.com/MjI1NzAz) [<img src="assets/imgsli3.png" height="150"/>](https://imgsli.com/MjI1NzAx) [<img src="assets/imgsli6.png" height="150"/>](https://imgsli.com/MjI1NzE2)



## <a name="news"></a> 🆕 News

- **2024-2-23:** This repo is released.
- **2024-2-23:** arXiv paper available.




## <a name="todo"></a> ☑️ TODO

- [ ] Build the repo
- [x] arXiv version
- [ ] Release code
- [ ] Pretrained weights
- [ ] More Tasks
 

## <a name="results"></a> 🥇 Results

We achieve state-of-the-art adaptation performance on various downstream image restoration tasks. Detailed results can be found in the paper.

<details>
<summary>Evaluation on Second-order Degradation (LR4&Noise30) (click to expand)</summary>

<p align="center">
  <img width="900" src="assets/SR&DN.png">
</p>
</details>


<details>
<summary>Evaluation on Classic SR (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/classicSR.png">
</p>
</details>


<details>
<summary>Evaluation on Denoise&DerainL (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/Dn&DRL.png">
</p>
</details>


<details>
<summary>Evaluation on Heavy Rain Streak Removal (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/DRH.png">
</p>
</details>


<details>
<summary>Evaluation on Low-light Image Enhancement (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/low-light.png">
</p>

</details>


<details>
<summary>Evaluation on Model Scalability (click to expand)</summary>

<p align="center">
  <img width="600" src="assets/scalabiltity.png">
</p>

</details>




## <a name="cite"></a> 🥰 Citation

Please cite us if our work is useful for your research.

```

```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at cshguo@gmail.com

