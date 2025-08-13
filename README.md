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

