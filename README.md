# Lab6

## 環境設定

### 虛擬環境建立與啟用

使用 Python 虛擬環境可避免套件衝突。請根據作業平台依序執行以下步驟：

- **MacOS / Linux:**

    ```bash
    python -m venv .venv
    source ./.venv/bin/activate

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

- **Windows**

    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate.bat

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

## 訓練

```bash
python -m src.train --epochs 200 --batch 128 --cond_dim 512
```


## 取樣

```bash
python -m src.sample --json data/test.json     --out images/test
python -m src.sample --json data/new_test.json --out images/new_test
```


## 評估

```bash
python -m src.evaluate --json data/test.json     --img_dir images/test
python -m src.evaluate --json data/new_test.json --img_dir images/new_test
```