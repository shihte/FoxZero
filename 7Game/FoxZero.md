排七 (Sevens) AlphaZero 演算法架構設計書

1. 狀態空間與特徵工程 (State Representation)

為了讓神經網絡能夠進行隱含推論（Implicit Inference），輸入張量（Input Tensor）必須包含「己方資訊」與「公共資訊」，嚴格排除上帝視角（他方手牌）。

張量維度設計： [Batch_Size, Channels, 4, 13]
將撲克牌分為 4 種花色 x 13 個點數。

Channel Index

特徵描述 (Feature Description)

數值意義

目的

0

己方手牌 (My Hand)

1=持有, 0=無

基礎決策依據

1

桌面已出牌 (Board State)

1=已出, 0=未出

判斷合法步與通道

2

合法步掩碼 (Legal Moves Mask)

1=可出, 0=不可

輔助網絡聚焦

3

玩家 1 (下家) 蓋牌數

Normalized (n/13)

推測威脅程度

4

玩家 2 (對家) 蓋牌數

Normalized (n/13)

推測威脅程度

5

玩家 3 (上家) 蓋牌數

Normalized (n/13)

推測威脅程度

6

玩家 1 (下家) 缺門紀錄

1=該花色已 Pass

關鍵推論特徵：推測斷牌位置

7

玩家 2 (對家) 缺門紀錄

1=該花色已 Pass

關鍵推論特徵：推測斷牌位置

8

玩家 3 (上家) 缺門紀錄

1=該花色已 Pass

關鍵推論特徵：推測斷牌位置

9

莊家標記 (Dealer Indicator)

全 1=我是莊家, 全 0=非莊

觸發風險趨避 (莊家輸兩倍)

10

歷史步數 (Turn Count)

Normalized (t/52)

調整策略積極度 (前期/後期)

註：缺門紀錄 (Channels 6-8) 是 AI 學會「擋牌」的關鍵。若上家在黑桃 Pass，網絡會學會扣住黑桃 7 之後的牌來實施打擊。

2. 神經網絡架構 (Network Architecture)

採用雙頭輸出（Two-Headed）的卷積神經網絡（CNN）或殘差網絡（ResNet）。由於排七狀態空間小於圍棋，建議使用輕量級架構以加速推論。

骨幹網絡 (Backbone)

輸入層：[C, 4, 13]

卷積層 (Conv Block)：256 Filters, 3x3 Kernel, Stride 1, Batch Norm, ReLU。

殘差塔 (ResNet Tower)：5 ~ 10 個 Residual Blocks（視硬體與收斂速度調整）。

每個 Block：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Skip Connection -> ReLU。

輸出頭 (Heads)

策略頭 (Policy Head)

目的：預測下一步出牌機率 $P(a|s)$。

結構：Conv1x1 (2 filters) -> BN -> ReLU -> Flatten -> Dense(52) -> Softmax。

輸出：52 維向量（對應 52 張牌）。注意：需與合法步掩碼 (Legal Mask) 做 Element-wise Multiply 後重新歸一化。

價值頭 (Value Head)

目的：預測當前局面最終勝率 $V(s)$。

結構：Conv1x1 (1 filter) -> BN -> ReLU -> Flatten -> Dense(256) -> ReLU -> Dense(1) -> Tanh。

輸出：純量 [-1, 1]。

3. 終局獎勵函數 (Terminal Reward Function)

此函數僅在遊戲結束（Game Over）時觸發。嚴格遵守零和博弈與倍率規則。

定義：

$P_i$: 玩家 $i$ 的手牌剩餘點數總和 (Raw Points)。

$is\_winner$: 布林值，是否為脫手者。

$is\_dealer$: 布林值，是否為莊家。

$base\_multiplier$: 基礎倍率 (脫手者出現，全員輸兩倍) = 2.0。

算法邏輯 (Python Pseudocode):

def calculate_final_rewards(players, dealer_index):
    """
    計算四位玩家的最終獎勵 (Normalized)。
    players: List of Player objects (包含手牌)
    """
    raw_penalties = []
    
    # 1. 找出贏家 (手牌數為 0 者)
    winner_index = next((i for i, p in enumerate(players) if len(p.hand) == 0), None)
    
    # 若流局 (極罕見)，視為無贏家，所有人皆計算罰分
    
    # 2. 計算每個人的罰分 (Penalty Calculation)
    for i, player in enumerate(players):
        if i == winner_index:
            raw_penalties.append(0.0)
            continue
            
        # 基礎點數
        score = sum(card.rank for card in player.hand) # A=1 ... K=13
        
        # 倍率堆疊 (Multipliers Stack)
        multiplier = 1.0
        
        # 規則 A: 有人脫手，輸家輸兩倍 (Global Multiplier)
        if winner_index is not None:
            multiplier *= 2.0
            
        # 規則 B: 莊家輸，輸兩倍 (Role Multiplier)
        if i == dealer_index:
            multiplier *= 2.0
            
        # 規則 C: 點數超過 50，輸兩倍 (Threshold Multiplier)
        if score > 50:
            multiplier *= 2.0
            
        # 計算最終罰分
        final_penalty = score * multiplier
        raw_penalties.append(final_penalty)

    # 3. 零和轉換 (Zero-Sum Transformation)
    # 贏家獎勵 = 所有輸家罰分的總和
    # 輸家獎勵 = 負的罰分
    
    rewards = []
    total_loss_pool = sum(raw_penalties)
    
    for i in range(4):
        if i == winner_index:
            rewards.append(total_loss_pool) # 正值
        else:
            rewards.append(-raw_penalties[i]) # 負值

    # 4. 數值歸一化 (Normalization)
    # 理論最大單人罰分估算：
    # 手牌約 90 分 * 2(脫手) * 2(莊家) * 2(破50) = 720 分
    # 為了讓 Tanh 活化函數工作良好，將分數壓回 [-1, 1] 區間
    SCALE_FACTOR = 800.0 
    normalized_rewards = [r / SCALE_FACTOR for r in rewards]
    
    return normalized_rewards


4. 訓練流程與優化 (Training Loop & Optimization)

MCTS 修正邏輯 (強制出牌規則)

由於規則限制「有牌必出」，決策樹的分支因子 (Branching Factor) 會大幅變動。

單選節點 (Single Move)：若合法步只有 1 種，跳過 MCTS 搜尋，直接執行該動作，但必須將此狀態 $(s, a)$ 存入 Buffer。這對於價值網絡學習「被迫蓋牌導致的惡果」至關重要。

多選節點 (Decision Node)：執行標準 MCTS (PUCT 算法)，模擬 800~1600 次。

損失函數 (Loss Function)

$$L = (z - v)^2 - \pi^T \log(p) + c \|\theta\|^2$$

$(z - v)^2$: 價值損失 (MSE)。讓預測勝率 $v$ 逼近最終獎勵 $z$。

特徵：因為有「破 50 分重罰」機制，$z$ 的分佈會呈現雙模態 (Bi-modal)，這會驅使 $v$ 在接近 40 分時急劇下降，形成風險趨避。

$-\pi^T \log(p)$: 策略損失 (Cross-Entropy)。讓神經網絡預測機率 $p$ 逼近 MCTS 搜尋後的機率分佈 $\pi$。

$c \|\theta\|^2$: L2 正則化，防止過擬合。

自我對局設置 (Self-Play Config)

探索雜訊 (Dirichlet Noise)：在根節點加入 $Dir(0.3)$ 雜訊，防止 AI 陷入單一固定的開局套路。

溫度參數 (Temperature)：

前 30 步：$\tau = 1.0$ (依機率探索)。

30 步後：$\tau \to 0$ (選擇訪問次數最多的動作，Greedy)。

非同步訓練：建議使用 1 個 GPU 進行訓練，CPU 多線程並行生成棋譜 (Actor-Learner 架構)。

5. 預期演化行為 (Expected Emergent Behaviors)

基於上述獎勵與輸入設計，模型將自行演化出以下策略：

精確斷頭 (Sniper Blocking)：

透過 Channels 6-8 (缺門紀錄)，AI 會發現：「上家在梅花 Pass -> 我扣住梅花 7 -> 上家罰分暴增 -> 我的 Reward 提升」。

莊家恐慌 (Dealer Panic)：

由於 Channel 9 (莊家標記) 與 2 倍罰分掛鉤，擔任莊家時 AI 會極度傾向打出大點數牌 (K, Q) 以逃避懲罰，即使這會幫助下家。

50分懸崖效應 (The 50-Point Cliff)：

當手牌分數累積至 40 分左右，AI 的 Value Network 輸出會崩跌。此時 AI 會放棄擋人，轉而優先打出最大點數的牌 (Cost Minimization)，避免觸發「破 50 輸兩倍」的毀滅性懲罰。