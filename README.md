# Probalilistic Latent Space Model implementation
## Usage
```
usage: main.py [-h] -m MODEL_PATH [-r TRAIN_RATIO] [-t NUM_OF_TOPIC] [-k TOPK] [-q QUERY_MODEL_PATH] [-v VALIDATION_MODEL_PATH]
               [-a ANS_PATH] [--ranking RANKING_LIST_DIR] [--test TEST_MODEL_PATH]

plsa

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH
  -r TRAIN_RATIO
  -t NUM_OF_TOPIC
  -k TOPK
  -q QUERY_MODEL_PATH
  -v VALIDATION_MODEL_PATH
  -a ANS_PATH
  --ranking RANKING_LIST_DIR
  --test TEST_MODEL_PATH
```
## Paper Reference
[Probabilistic Latent Semantic Analysis](https://arxiv.org/pdf/1301.6705.pdf)
## Model Research
[Note For Probabilistic Latent Semantic Indexing](https://hackmd.io/GWjbRr_mRrmvdturVgQZjg)
## Methodology
演算法不複雜，但是實作上比較困難的有兩點
* 如何知道 EM 實作是正確的
* 當 model 的表現質量很差時，如何用數字去解釋，找出根源
### 如何知道 EM 實作是正確的
利用了 EM 非常重要的特質，就是 Likelihood 永遠會朝著 Optimal 前進。因此只要當 Likelihood 下降時，就知道 EM 實作錯了。當 Likelihood 都朝著 Optimal 前進，有很高的機會實作是正確的。果然在查看結果時，每個 Topic 的 Term 質量還不錯。
### 當 model 的表現質量很差時，如何用數字去解釋，找出根源。
實驗的模型普遍在 Term 上面分得還不錯，但是 Document 在一些 Dataset 的表現很不穩定。

我發現分的好的 Dataset ，在每個 ``P(z|d)`` 的 variance 都很高，也就是每個文件屬於哪一個主題基本上是很明顯的。另一方面，表現比較不好的 Dataset ， ``P(z|d)`` 的 variance 就相對低。換句話說，一個文件的前三個 Topic ，機率可能很接近。 

透過觀察 ，才發現原來表現不如預期的 dataset 過早停止了 ``EM`` algorithm，導致 underfitting。調整了 Early stop 的條件後，成功改善了分主題的質量。

舉例來說，在 cdn_loc_0000998 這個文件中，28 個 iterations 後的 variance 為相當高的 0.09，而 15 個 iterations 的僅有 0.052。觀察 Term 後發現，只有 15 個 Iteration 的在黑面琵鷺這個主題，把「合併」當成第一名，顯示模型仍然在 underfitting 的狀態。經過更多的 iterations 後，前五名的 Term 非常漂亮，也在總體的 MAP 有顯著的成長。

| Doc_id          | Title                | var(P(z\|d)) EM 28 iterations | var(P(z\|d)) EM 15 iterations |
|-----------------|----------------------|-------------------------------|-------------------------------|
| cdn_loc_0000998 | 曾文溪口黑面琵鷺來了 | 0.09                          | 0.052                         |

| 鷺     | 琵     | 琵鷺    | 面琵    | 黑面    |
|--------|--------|---------|---------|---------|
| 0.0523 | 0.0505 | 0.04896 | 0.04896 | 0.04857 |

| 合併 | 琵     | 琵鷺   | 面琵   | 鷺     |
|------|--------|--------|--------|--------|
| 0.1  | 0.0322 | 0.0319 | 0.0319 | 0.0316 |

|     | EM 28 iterations | EM 15 iterations |
|-----|------------------|------------------|
| MAP | 0.527535         | 0.39804          |

## Experient Result

### Topic 分類的質化表現
我們在隨機抽樣 20 篇文件的 Corpus 上，發現效果非常顯著。表格是顯示其中一個 Topic 裡面，Term 的分佈，雖然有些字只是不成詞的 Bigram ，仍然可以看出這個 Topic 與金融市場非常相關。在檢視 ``P(z|d)`` 時， cts_eco_0000267 這一個 Topic 的組成比例很高，他的標題是「股市回挫，債市回溫」，與 Topic 的相關性很高。統計了每個 Doc 主題分布的 vairance，平均約莫是 0.09，也就是 Document 的主題很明確，相關的主題機率分佈就高，不相關的機率就低。

| Term    | 債券  | 型基  | 買回  | 附買  | 券型  | 均成  | 客臨  | 債市  | 轉到  | 小柯  |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| P(w\|z) | 0.166 | 0.065 | 0.065 | 0.056 | 0.046 | 0.046 | 0.037 | 0.037 | 0.037 | 0.028 |

| doc id          | 標題              | P(z\|d) |
|-----------------|-------------------|---------|
| cts_eco_0000267 | 股市回挫 債市升溫 | 0.99999 |

### MAP 的表現
我們在第一次作業 ans_train.csv 的 Corpus 做訓練，以 query-train 當作 Query 的 Input。我們做出來的結果是 PLSI 在 MAP 僅有 0.5275 的表現，遠遠不及 LSI。我們觀察到雖然在 Training document 的 variance 為 0.071 ，但是在 Query 的 variance 卻很差，僅有 0.053。 

|     | PLSI     | LSI  |
|-----|----------|------|
| MAP | 0.527535 | 0.93 |

以 query_id 001 為例，Topic 6 的機率最高，也確實在 Topic 6 單詞的分佈可以看出這個主題與流浪犬有關。然而，topic 0 和 topic 1 也佔了相當的比例，導致機率分佈的 variance 為偏低的 0.03。觀察後發現 Topc 0 和 Topic 1 可能都與學制相關。

| query_id | title      | MAP      | var(P(z\|q)) | topic 6 | topic 0 | topic 1 |
|----------|------------|----------|--------------|---------|---------|---------|
| 001      | 流浪狗問題 | 0.336477 | 0.03         | 0.6047  | 0.1052  | 0.1028  |

| Topic id | 狗   | 犬     | 卡特   | 流浪   | 浪狗  |
|----------|------|--------|--------|--------|-------|
| 6      | 0.06 | 0.0452 | 0.0394 | 0.0351 | 0.019 |

| Topic id | 合併   | 入學  | 推薦   | 甄    | 薦甄   |
|----------|--------|-------|--------|-------|--------|
| 0        | 0.0512 | 0.021 | 0.0198 | 0.018 | 0.0163 |

| Topic id | 入學  | 聯招  | 招分  | 考招   | 測驗   |
|----------|-------|-------|-------|--------|--------|
| 1        | 0.065 | 0.049 | 0.042 | 0.0416 | 0.0407 |

對比 query_004，Topic 4 的機率為 0.999，variance 高達 0.899，其 MAP 為所有 query 裡表現第二好的 0.75。然而這只是少數的例子，多數的 var(P(z|q)) 落在 0.4 左右

| query_id | title        | MAP  | var(P(z\|q)) | topic 4 | topic 5 | topic 1 |
|----------|--------------|------|--------------|---------|---------|---------|
| 004      | 台灣黑面琵鷺 | 0.75 | 0.089        | 0.9999  | 2e-11   | 1e-15   |

| Topic id | 鷺    | 琵    | 琵鷺  | 面琵  | 黑面  |
|----------|-------|-------|-------|-------|-------|
| 4        | 0.064 | 0.061 | 0.059 | 0.059 | 0.059 |

這意味著 PLSI 雖然可以在看過的資料上找到對應的 Term ，且在看過的文件上能做出很清楚的分類，但面對陌生的 Query ，很難生成精準的 P(z|q)。說明了我們實驗中的模型確實有 overfitting 的問題，因此產生的 MAP 表現遠遠不及 LSI








### Testing
面對沒看過的文件，PLSI 使用的 Document Folding 假設了同一個主題會使用同一個 Topic distribution，並利用 EM 來學 P(z|d)。然而，由於我們不清楚新的文件主題分佈為何，因此會給一個隨機的值，EM 則會根據這個隨機的值，走到 Local Optimal。

然而，這一個隨機的值，有可能會讓 P(z|d) 走向意想不到的方向。舉例來說，我們的模型在英語師資這個主題選出了很精準的 Term 。

| Topic id | 英語   | 國小   | 語教  | 師資  | 測驗   |
|----------|--------|--------|-------|-------|--------|
| 1        | 0.1107 | 0.0419 | 0.041 | 0.039 | 0.0388 |


然而當我們丟給模型一個沒學過的文件時，有時會產生意外的主題分佈。以 cdn_edu_0001104 為例，其標題為「國小英語師資需經檢訓合格」，且在文中多次出現英語，國小，師資等關鍵字。然而當我們去檢查 Document 的 topic distribution，Topic 1 卻沒有在前五名，反而是 Topic 0 佔了 43 %，一個混雜著國民卡和海相關的主題。

| doc id          | 標題                     | Topic 0 | Topic 9 | Topic 4 |
|-----------------|--------------------------|---------|---------|---------|
| cdn_edu_0001104 | 國小英語師資需經檢訓合格 | 0.433   | 0.13    | 0.12    |

| Topic id | 民卡  | 業稅  | 隱私  | 珠海   | 漁會   |
|----------|-------|-------|-------|--------|--------|
| 0       | 0.021 | 0.017 | 0.015 | 0.0127 | 0.0115 |

總體來看，PLSA 在 Testing dataset 裡所生成 P(z|d) 的 variance 平均下來是 0.01，相當不理想。

面對 Query 的狀況類似，對於模型來說，也是新的 Document，因此也能解釋為何 Query 很難生成精準的 P(z|d)。

LDA 利用 Dirichlet Distribution 去產生 P(w|z) 和 P(z|d)，因此避免了 P(z|d) 選到不好的初始值，以及訓練完就永遠固定的 P(w|d)，試圖達到更好的 generalization。 




## Conclusion
PLSA 是 LSA 的機率版本，有更好的理論保證，且在學到的參數有直觀的解釋。不管對於隨機 Sample 的 Dataset，抑或是 ans-train.csv 裡的 Corpus，利用 P(w|z) 選出重要的 Term 都可以很清楚解讀 Topic 的方向。

然而，因為缺乏合適的 P(z|d) 初值，因此對於新的文件或是 Query ，都很難生成 var P(z|d) 較大，較精準的 P(z|d)，加上參數過多導致的 overfitting ，導致本次實驗在 ans-train.csv 裡的 Corpus IR MAP ， 抑或是給定一個新的文件，去生成主題分佈的效果皆不好。

