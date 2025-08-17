# Abstract
Continual urban expansion is driving demand for simulations of crowd-movement to support traffic management and emergency response. However, these efforts are hindered by a computational dilemma, wherein single general-purpose models are unable to capture the heterogeneity across urban regions and situations, while modeling areas independently incurs high computational overhead and fails to account for spatial correlations. This study developed a novel Tree-structured Diffusion Model for the modeling of crowd distributions, integrating decision trees for data partitioning with generative diffusion models. The LightGBM model is first used to analyze spatiotemporal data to identify key factors influencing crowd flow and create an hierarchy of regions (root, branches, leaves). A diffusion model is then pretrained at the root for the dissemination of knowledge to branches and leaf models to enable fine-tuning for specific data subsets. This approach reduces the required data volumes, enabling the training and simulation of specific scenarios that would otherwise be difficult to model independently. The validity of the framework was assessed using a real-world Taipei City crowd-flow dataset. The leaf (expert) model outperformed a conventional single diffusion (baseline) model and high-level branch models across standard evaluation metrics. These findings demonstrate the efficacy of the proposed framework in managing data heterogeneity and achieving a balance between computational efficiency and simulation accuracy for complex, city-scale crowd modeling.</br>
</br>Keywords: Crowd Distribution Prediction, Decision Tree Model, Diffusion Model, Tree-structured Diffusion Model, Transfer Learning, Spatio-temporal Data, Feature Engineering
 
# 摘要
近年來，隨著都市化發展，大範圍的人群移動模擬對於交通管理與災害應變至為關鍵。然而，現有的深度學習方法普遍面臨一個兩難困境：採用單一通用模型，難以捕捉城市中不同區域與事件的異質性；而為各區域獨立建模，則計算成本過於高昂且忽略了空間關聯性。為解決此問題，本研究提出一個創新的「樹狀擴散模型(Tree-like Diffusion Model)」框架，旨在結合決策樹的數據劃分能力與擴散模型的強大生成能力，以高效率生成高保真度的人群分布。首先，本框架利用 LightGBM 模型對時空數據進行分析，自動找出影響人流的關鍵因子，並據此建構一個階層式的數據分群結構。接著，我們採用預訓練與遷移學習的策略：先以完整的城市數據訓練一個通用的根模型(Root Model)作為骨幹，再將其知識遷移至針對特定數據子集進行微調的分支模型(Branch Model)與葉模型(Leaf Model)。此舉大幅降低了對數據量的要求，使得那些樣本數較少、難以獨立建模的特定情境(如特殊事件)能夠被成功地訓練與模擬，顯著提升了訓練資料的利用效率。本研究以真實世界的台北市人群流動數據集進行驗證，實驗結果表明，本研究提出的樹狀擴散框架，其專家模型(Leaf Model)在各項評估指標上，均顯著優於傳統的單一擴散模型(Baseline Model)以及較為通用的上層模型(Branch Model)。此結果驗證了本框架能有效處理數據異質性，並在計算效率與模擬精度之間取得優異的平衡，為複雜的都市人群模擬任務提供了一條新的解決途徑。</br>
</br>關鍵字：人群分布預測、決策樹模型、擴散模型、樹狀擴散模型、遷移學習、時空資料、特徵工程

# 程式碼執行順序
1. 資料集在Taipei_CF，一般來說直接使用all_merged.csv即可
1. 執行decisionTree_lightgbm.py跑特徵決策樹跟特徵分群
1. 執行DDPM_Long-term.py生成第一階段人群分布結果
1. 執行DDPM_Long-term_two-stage.py生成第二階段人群分布結果
1. 執行DDPM_Long-term_3stage.py生成第三階段人群分布結果
1. 執行analyze_stage3_error.py檢視第三階段誤差情形，可再執行decisionTree_lightgbm.py分群
1. 執行DDPM_Long-term_Baseline.py生成比較基準人群分布結果
1. 執行DDPM_Long-term_4stage.py生成第四階段人群分布結果

剩下程式碼已廢棄不用</br>
DDPM_3DXXX為時序+空間的人群分布模擬模型

