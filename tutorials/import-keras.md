# 如何將 Keras 模型導入 TensorFlow.js

Keras 模組（通常透過 Python API 創建）的多種格式可以壓縮成一文件儲存，而「整個模型」的格式可以轉換成 TensorFlow.js Layers 格式，並且可以直接加載到 TensorFlow.js 中進行推理或進一步的訓練。

TensorFLow.js Layers 格式是一個目錄包含 `model.json` 文件以及一組二進位制共享權重的文件，而 `model.json` 文件包含拓樸模型（又名「結構」、「圖形」：層的描述以及它們如何連接）及權重文件的清單。

## 要求

轉換的過程需要 Python 的環境，而你可以使用 [pipenv](https://github.com/pypa/pipenv) 或 [virtualenv](https://virtualenv.pypa.io/en/stable/) 來保持一個獨立的環境。若要安裝轉換器，請使用`pip install tensorflow`。

將 Keras 模型導入 TensorFlow.js 是一個兩步驟的程序，首先，將現有的 Keras 模型轉換成 TensorFlow.js Layers 格式，然後再將其加載到 TensorFlow.js 中