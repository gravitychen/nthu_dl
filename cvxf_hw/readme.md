# NTHU_CVFX_HomeWork2_Team20



# Outline

**1.Training MUNIT**

**2.Other three methods**

**3.Inference result**

**4.Compare&Conclusion**

## 1.Training MUNIT



training logs:

<img src="https://i.imgur.com/nhklRrB.png" width="355">


saved model:

<img src="https://i.imgur.com/noItfQj.png" width="355">





## **2. Other three methods**
> 1. DRIT
> 2. FastPhotoStyle
> 3. neural_style


- ### 2.1 DRIT

model:
![](https://i.imgur.com/iNuJ7nF.png)


DRIT用GAN學兩個domain X和Y 之間的映射關係.受CycleGAN和MUNIT的啓發，CycleGAN的兩個latent space是分開的，MUNIT共享latent space. DRIT則共享content space，獨享attribute space.Network同樣由cross-domain部分和within domain兩個部分組成，cross-domain用來生成交換風格後的圖像，within domain用來重建圖像，同時也使用Cycle Consistency Loss.

### 2.2 [FastPhotoStyle](https://arxiv.org/abs/1802.06474)       ------ECCV 2018

NVIDIA和加州大學的研究團隊提出了一種新的解決照片級圖像風格化的方法——FastPhotoStyle。該方法包括兩個步驟：風格化和平滑化。大量的實驗表明，該方法生成的圖像比以前的最先進的方法更真實、更引人注目。更重要的是，由於採用封閉式的解決方案，FastPhotoStyle生成風格化圖像的速度比傳統方法快49倍。

![](https://i.imgur.com/oX0STp8.png)
可以看到FastPhotoStyle的network算是比較輕量級的。model由stylization step和smoothing step組成。當stylization step將引用照片的樣式轉換為內容照片時，smoothing step確保空間上一致的樣式化。

### 2.3 neural_style

![](https://i.imgur.com/7HKckti.png)

這是傳統的做法，x通過fw神經網路生成圖片y,再把content圖像（即x）和style圖像在通過VGG後取得的不同層的特徵圖，並以此來衡量和限制生成圖片y，使之不僅能保留content的結構，並且具有style影像的紋理風格。




# **3.Inference result**


|    MUNIT                                                | ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/nCqHRiW.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/HQgjkUj.png)                                    | ![](https://i.imgur.com/4HwsI3E.png)                                    | ![](https://i.imgur.com/ab7PnUQ.png)                                    |
| ![](https://i.imgur.com/Cj3hdo3.png) | ![](https://i.imgur.com/78ZbIQ0.png)                                   | ![](https://i.imgur.com/KbU9q7t.png)                                   | ![](https://i.imgur.com/jwnZFbx.png)                                  |
| ![](https://i.imgur.com/jXL2sqg.png)| ![](https://i.imgur.com/oxWhjjx.png)                                  | ![](https://i.imgur.com/H0aZXj3.png)                                  | ![](https://i.imgur.com/PffnLjS.png)                                 |

以上是將古風轉換為照片的inference的結果，可以看到當照片的顏色偏暗的話，結果一樣會變暗，反之亦然。可以看到結果並不是那麼理想，因為比如其中的人像，經過轉換之後還有看得出會有人像的部分。但是時間在訓練過程中，效果都挺好的，但是不知道為什麼inference的時候效果並不理想。

|    MUNIT                                                          | ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/Cj3hdo3.png) | ![](https://i.imgur.com/jXL2sqg.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/ico3aiO.png)                                    | ![](https://i.imgur.com/aXmm4iO.png)                                    | ![](https://i.imgur.com/t1oNztm.png)                                   |
| ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/YgehFw2.png)                                   | ![](https://i.imgur.com/U1cuDSm.png)                                   |  ![](https://i.imgur.com/Ls3pt8w.png)                                  |
| ![](https://i.imgur.com/nCqHRiW.png) | ![](https://i.imgur.com/90cLh1h.png)                                  | ![](https://i.imgur.com/0to7330.png)                                  | ![](https://i.imgur.com/PEhRQKd.png)                                 |

當使用model進行照片轉到古畫的時候，效果就好多了。可以看到九張圖的顏色整體都會偏到淡黃色，當要轉換的圖片風格偏藍(綠)色的時候，轉換后一樣出現偏藍(綠)色。

----
----

### **3.1 DRIT**


| DRIT                                 | ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/Cj3hdo3.png) | ![](https://i.imgur.com/jXL2sqg.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/4aahfP2.png)                                    | ![](https://i.imgur.com/ECaesN2.png)                                    | ![](https://i.imgur.com/1FS6oAS.png)                                   |
| ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/h7VbKlD.png)                                   | ![](https://i.imgur.com/N8FR84I.png)                                   | ![](https://i.imgur.com/9B3Zp8l.png)                                  |
| ![](https://i.imgur.com/nCqHRiW.png) | ![](https://i.imgur.com/HcHmjuB.png)                                  | ![](https://i.imgur.com/LpFFUo8.png)                                  | ![](https://i.imgur.com/2HH8org.png)                                 |



| DRIT                                 | ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/nCqHRiW.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/jXL2sqg.png) | ![](https://i.imgur.com/J8lMD3c.png)| ![](https://i.imgur.com/uAlhNZO.png)                                    | ![](https://i.imgur.com/SgUh2At.png)                                   |
| ![](https://i.imgur.com/Cj3hdo3.png) | ![](https://i.imgur.com/s8Wf0vl.png)                                   | ![](https://i.imgur.com/JQXw1VZ.png)                                   | ![](https://i.imgur.com/d3iBGUh.png)                                  |
| ![](https://i.imgur.com/DNsRM0v.png) |  ![](https://i.imgur.com/VCcwEuu.png)                          | ![](https://i.imgur.com/3qzZCmR.png)                                    | ![](https://i.imgur.com/oibFdd9.png)


使用DRIT按這個方法，我認為在古風的方面，它的效果會比較好，它展現出來的畫更生動，比較少出現artifact，但是當使用古畫轉到照片的時候，其中第二張圖可以很明顯的看出，轉換后的結果還是一個畫畫中的女子。



----
----

### **3.2 FastPhoto-Style**

| FastPhoto-Style                        | ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/Cj3hdo3.png)|![](https://i.imgur.com/jXL2sqg.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/klgcr9G.png) |![](https://i.imgur.com/PEJfajr.png)| ![](https://i.imgur.com/qBt32VL.png)| ![](https://i.imgur.com/0Ky7fCh.png)
| ![](https://i.imgur.com/cEDUBRZ.png) |![](https://i.imgur.com/g6Zr1WK.png)|![](https://i.imgur.com/sWcDiyj.png)| ![](https://i.imgur.com/rLbYHiC.png)|
| ![](https://i.imgur.com/nCqHRiW.png) | ![](https://i.imgur.com/0PUMzVc.png)| ![](https://i.imgur.com/znoV8h4.png)| ![](https://i.imgur.com/c3nAGfw.png)|

fast-photo-style這個方法 能較好的均衡顏色方面的信息，可以看到當第二張要轉換的圖像上的衣服顏色為紅藍相間，所以三張content轉換之後的成果都表現出了紅色與藍色。

| FastPhoto-Style              | ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/nCqHRiW.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
|![](https://i.imgur.com/DNsRM0v.png) |![](https://i.imgur.com/9iBrvfU.png)|![](https://i.imgur.com/XOQBm3y.png)| ![](https://i.imgur.com/ezI6sij.png)|
| ![](https://i.imgur.com/Cj3hdo3.png) |![](https://i.imgur.com/Q2lOrzG.png)|![](https://i.imgur.com/fbsQb5g.png)|![](https://i.imgur.com/h3IXLkc.png)
|![](https://i.imgur.com/jXL2sqg.png)|![](https://i.imgur.com/uPVVZpr.png)|![](https://i.imgur.com/E6HKkpZ.png)|![](https://i.imgur.com/yJ0wGAL.png)|



----
----


### **3.3 neural-style**



| neural-style                                 | ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/Cj3hdo3.png) | ![](https://i.imgur.com/jXL2sqg.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/ShYcXDV.png)| ![](https://i.imgur.com/ZRaCKV3.png)| ![](https://i.imgur.com/Sxs4t2P.png)
| ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/z6qoQiJ.png)| ![](https://i.imgur.com/VGQwlwt.png)| ![](https://i.imgur.com/THAS58N.png)
| ![](https://i.imgur.com/nCqHRiW.png) | ![](https://i.imgur.com/4F0jzll.png)| ![](https://i.imgur.com/wdbiWyy.png)| ![](https://i.imgur.com/GZ2ID9P.png)|




| neural-style                                | ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/nCqHRiW.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/DNsRM0v.png) |![](https://i.imgur.com/bNsneNP.png)| ![](https://i.imgur.com/zcDYd3Y.png)| ![](https://i.imgur.com/QAMcDEp.png)|
| ![](https://i.imgur.com/Cj3hdo3.png) |![](https://i.imgur.com/51yxt7O.png)|![](https://i.imgur.com/T5XYCDY.png)| ![](https://i.imgur.com/jMZAeHR.png)
|![](https://i.imgur.com/jXL2sqg.png) |![](https://i.imgur.com/Ewx5Tay.png)|![](https://i.imgur.com/trgp5L7.png)|![](https://i.imgur.com/nKabh6K.png)|

neural-style 這個方法比較心狠手辣，雖然也保留著一點原來的content，但是相比較其他方法，content破壞了許多。特別是在照片轉古畫的時候，幾乎看不出原圖有什麼東西。



----
----

## Compare&Conclusion

|               | MUNIT | DRIT | FastPhoto-Style | neural-style | Cycle-GAN |
|---------------|-------|------|-----------------|--------------|-----------|
| 訓練耗時      |   很耗時    |  很耗時    |      迅速           |    3mins          |       很耗時    |
| inference時間      |  迅速     |  迅速    |         快        |     快     |     快      |
| inference效果 |    一般   |   差   |        差         |          一般    |      佳     |
| GPU使用       |   使用    |  使用    |       使用          |   使用          |   使用      |
| 使用的memory  |  4G     |   10G(bs=128)    |      1G |    4G  |      5G      |


Conclusion:
   可以看到不管用什麼方法， 照片轉到古畫的時候，表現都會比較好，我想是因為古畫的特征之一就是背景都是淡黃色的顏色。model學習的內容既然就不需要像古風轉現實那麼多，畢竟現實的image有千千萬萬種顏色，model也比較難以駕馭。



<br/><br/>
<br/><br/><br/>
