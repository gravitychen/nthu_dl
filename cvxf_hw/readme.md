
# NTHU_CVFX_HomeWork2_Team20



# Outline

**1.Training MUNIT
2.Other four methods
3.Inference result
4.Compare&Conclusion**

## 1.Training MUNIT



training logs:
<img src="https://i.imgur.com/nhklRrB.png" width="355">


saved model:
<img src="https://i.imgur.com/noItfQj.png" width="355">





## **2. Other four methods**
> 1. DRIT
> 2. FastPhotoStyle
> 3. neural_style
> 4. cycle-GAN

- ### DRIT

model:
![](https://i.imgur.com/iNuJ7nF.png)


DRIT用GAN學兩個domain X和Y 之間的映射關係.受CycleGAN和MUNIT的啓發，CycleGAN的兩個latent space是分開的，MUNIT共享latent space. DRIT則共享content space，獨享attribute space.Network同樣由cross-domain部分和within domain兩個部分組成，cross-domain用來生成交換風格後的圖像，within domain用來重建圖像，同時也使用Cycle Consistency Loss.

### FastPhotoStyle

### neural_style

### cycle-GAN





# **3.Inference result**


|    MUNIT                                                | ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/nCqHRiW.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/HQgjkUj.png)                                    | ![](https://i.imgur.com/4HwsI3E.png)                                    | ![](https://i.imgur.com/ab7PnUQ.png)                                    |
| ![](https://i.imgur.com/Cj3hdo3.png) | ![](https://i.imgur.com/78ZbIQ0.png)                                   | ![](https://i.imgur.com/KbU9q7t.png)                                   | ![](https://i.imgur.com/jwnZFbx.png)                                  |
| ![](https://i.imgur.com/jXL2sqg.png)| ![](https://i.imgur.com/oxWhjjx.png)                                  | ![](https://i.imgur.com/H0aZXj3.png)                                  | ![](https://i.imgur.com/PffnLjS.png)                                 |


|    MUNIT                                                          | ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/Cj3hdo3.png) | ![](https://i.imgur.com/jXL2sqg.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/klgcr9G.png) | ![](https://i.imgur.com/ico3aiO.png)                                    | ![](https://i.imgur.com/aXmm4iO.png)                                    | ![](https://i.imgur.com/t1oNztm.png)                                   |
| ![](https://i.imgur.com/cEDUBRZ.png) | ![](https://i.imgur.com/YgehFw2.png)                                   | ![](https://i.imgur.com/U1cuDSm.png)                                   |  ![](https://i.imgur.com/Ls3pt8w.png)                                  |
| ![](https://i.imgur.com/nCqHRiW.png) | ![](https://i.imgur.com/90cLh1h.png)                                  | ![](https://i.imgur.com/0to7330.png)                                  | ![](https://i.imgur.com/PEhRQKd.png)                                 |


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
|




### **3.2 FastPhoto-Style**

| FastPhoto-Style                        | ![](https://i.imgur.com/DNsRM0v.png) | ![](https://i.imgur.com/Cj3hdo3.png)|![](https://i.imgur.com/jXL2sqg.png) |
|--------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------|
| ![](https://i.imgur.com/klgcr9G.png) |![](https://i.imgur.com/PEJfajr.png)| ![](https://i.imgur.com/qBt32VL.png)| ![](https://i.imgur.com/0Ky7fCh.png)
| ![](https://i.imgur.com/cEDUBRZ.png) |![](https://i.imgur.com/g6Zr1WK.png)|![](https://i.imgur.com/sWcDiyj.png)| ![](https://i.imgur.com/rLbYHiC.png)|
| ![](https://i.imgur.com/nCqHRiW.png) | ![](https://i.imgur.com/0PUMzVc.png)| ![](https://i.imgur.com/znoV8h4.png)| ![](https://i.imgur.com/c3nAGfw.png)|


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




----
----

## Compare&Conclusion

|               | MUNIT | DRIT | FastPhoto-Style | neural-style | Cycle-GAN |
|---------------|-------|------|-----------------|--------------|-----------|
| 訓練耗時      |   很耗時    |  很耗時    |      迅速           |    3mins          |       很耗時    |
| inference時間      |  迅速     |  迅速    |         快        |     快     |     快      |
| inference效果 |    1   |   2   |        差         |          一般    |      佳     |
| GPU使用       |   1    |  2    |       0.5G          |   6G           |   1.7       |
| 使用的memory  |  佔用大量記憶體     |   佔用大量記憶體    |      1G |      |      1.7G      |

Conclusion:






