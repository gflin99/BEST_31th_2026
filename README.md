# Microbial Fermentation Trend Prediction

## 專案目的
本專案旨在利用深度學習時間序列模型，針對微生物醱酵過程中的關鍵指標：**生物量 (OD600)、乳酸濃度 (LA) 進行下一時刻預測。

專案中實作並比較了四種神經網路架構：
1. **AE-GRU** (Autoencoder 結合 Gated Recurrent Unit)
2. **AE-LSTM** (Autoencoder 結合 Long Short-Term Memory)
3. **vinalla GRU**
4. **vinalla LSTM**

透過建立穩健的模型，我們能夠有效捕捉醱酵槽內的非線性動態變化，為未來的生物製程優化與自動化控制提供可靠的預測基礎。

## 環境需求
本專案基於 Python 3 開發，核心依賴套件如下：
- TensorFlow / Keras (包含對 GPU 加速的支援)
- Pandas & NumPy (資料處理)
- Scikit-learn (特徵縮放 MinMaxScaler 與模型評估)

安裝依賴套件：
```bash
pip install -r requirements.txt
