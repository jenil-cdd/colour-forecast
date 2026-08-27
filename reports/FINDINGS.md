# Findings — duvet cover set colour/size cold-start forecasting

Training data through **2026-05-31**; held out **2026-06-01 .. 2026-07-31**.


## 1. Model leaderboard, single held-out window

```
                      cold_wape  cold_mae  cold_bias  cold_spearman  cold_top6_hit  cold_coverage80
poisson_glm               0.597    55.929     -0.354          0.382          0.167            0.789
negbin_glm                0.607    56.910     -0.387          0.229          0.167            0.737
naive_family              0.607    56.928     -0.157          0.585          0.833            0.684
lasso                     0.636    59.654     -0.419          0.135          0.333            0.579
hier_bayes                0.648    60.768      0.133          0.507          0.500            0.947
knn_lookalike             0.652    61.116      0.100          0.453          0.333            0.632
elasticnet                0.656    61.492     -0.403          0.093          0.333            0.579
gompertz                  0.660    61.842     -0.306          0.106          0.167            0.632
logistic                  0.668    62.576     -0.323          0.082          0.167            0.632
bass                      0.672    62.949     -0.328          0.081          0.167            0.579
random_forest             0.680    63.763      0.002          0.155          0.333            0.737
ensemble                  0.686    64.271      0.142          0.475          0.333            0.737
heuristic                 0.725    67.915     -0.679          0.385          0.500            0.842
ridge                     0.745    69.806     -0.345         -0.184          0.167            0.421
lightgbm                  0.752    70.462      0.396          0.385          0.500            0.421
matrix_factorisation      0.752    70.522     -0.257          0.096          0.167            0.579
cluster_pool              0.780    73.120     -0.173          0.182          0.167            0.842
size_ratio                0.808    75.722      0.119          0.517          0.667            0.842
xgboost                   0.850    79.678      0.139         -0.083          0.333            0.421
```

## 2. Rolling-origin validation across 7 launch events

```
                      origins  wape_median  wape_mean  wape_iqr  mean_rank  worst_rank  spearman_median  top6_hit  coverage80  bias_median
model                                                                                                                                     
knn_lookalike               7        0.485      0.780     0.321      4.000         8.0            0.465     0.667       0.604        0.030
negbin_glm                  7        0.636      0.936     0.070      6.000        11.0            0.079     0.667       0.408       -0.429
hier_bayes                  7        0.675      1.178     0.147      6.286        17.0            0.500     0.667       0.645       -0.061
poisson_glm                 7        0.647      0.965     0.069      6.714        12.0           -0.039     0.643       0.496       -0.414
matrix_factorisation        7        0.736      0.897     0.170      7.857        12.0            0.399     0.667       0.356       -0.459
random_forest               7        0.730      0.908     0.315      8.286        16.0            0.254     0.595       0.589       -0.187
ensemble                    7        0.679      1.128     0.205      8.857        15.0            0.400     0.667       0.523       -0.450
heuristic                   7        0.762      0.815     0.083      8.857        15.0            0.398     0.667       0.610       -0.658
lasso                       7        0.684      1.035     0.280      9.000        13.0            0.414     0.643       0.259       -0.557
elasticnet                  7        0.710      0.993     0.185      9.714        13.0            0.400     0.643       0.272       -0.449
cluster_pool                7        0.758      0.990     0.201     10.429        15.0            0.060     0.619       0.577       -0.467
xgboost                     7        0.777      1.138     0.324     11.714        19.0           -0.078     0.643       0.483       -0.506
gompertz                    7        0.867      1.123     0.648     11.857        19.0            0.500     0.667       0.224       -0.315
bass                        7        0.855      1.190     0.600     12.286        17.0            0.441     0.643       0.208       -0.334
logistic                    7        0.858      1.170     0.609     12.286        18.0            0.441     0.667       0.216       -0.325
lightgbm                    7        0.744      1.279     0.197     12.571        17.0            0.190     0.643       0.343       -0.492
size_ratio                  7        0.834      2.901     3.641     13.000        19.0            0.500     0.690       0.711        0.387
ridge                       7        0.841      1.541     0.749     14.143        18.0            0.400     0.643       0.189       -0.250
naive_family                7        1.346      3.759     4.466     16.143        19.0            0.538     0.714       0.641        0.881
```

## 3. Order fractile sensitivity vs realised demand

Actual 120-day-equivalent cohort demand: **4,435 units**. Cost index uses stockout:holding = 0.9:0.1.

```
 fractile  total_order  vs_actual  units_short  units_excess  cost_index
      0.4         3643       0.82         1734           942        1655
      0.5         4459       1.01         1381          1404        1383
      0.6         5457       1.23         1036          2058        1138
      0.7         6774       1.53          779          3118        1013
      0.8         8724       1.97          416          4704         845
      0.9        12390       2.79            0          7955         795
```

- perfect-hindsight cost index: 0
- incumbent heuristic (2,180 units, flat split): 2,271

## 4. Recommended order sheet

```
           colour  size            program shade_family  family_depth_entered  forecast_120d   p10   p50    p90  return_rate  recommended_order  order_if_sized_per_sku  actual_units_in_test_window  actual_exposure_days
           Silver  Twin 400 TC Duvet Cover   Light Grey                   2.0          640.0 144.0 508.0 1782.0          0.1             1124.0                  1177.0                         69.0                    61
    Antique White  Twin 400 TC Duvet Cover   Near White                   2.0          543.0 122.0 431.0 1513.0          0.1              959.0                  1005.0                        240.0                    61
       Light Grey  Twin    Duvet Cover Set   Light Grey                   1.0          515.0 116.0 409.0 1435.0          0.1              905.0                   948.0                         64.0                    61
Indigo Dusty Blue  Twin    Duvet Cover Set     Mid Blue                   1.0          414.0  93.0 328.0 1153.0          0.1              728.0                   763.0                         75.0                    61
    Antique White Queen 400 TC Duvet Cover   Near White                   2.0          337.0  76.0 268.0  940.0          0.2              600.0                   629.0                        318.0                    61
            Taupe  Twin 400 TC Duvet Cover        Taupe                   0.0          313.0  70.0 248.0  872.0          0.1              550.0                   576.0                         66.0                    61
    Antique White  King 400 TC Duvet Cover   Near White                   2.0          285.0  64.0 226.0  796.0          0.2              511.0                   536.0                        276.0                    61
      Silver Gray Queen 400 TC Duvet Cover   Light Grey                   1.0          282.0  63.0 224.0  787.0          0.1              498.0                   522.0                         93.0                    60
           Silver  King 400 TC Duvet Cover   Light Grey                   1.0          249.0  56.0 197.0  693.0          0.1              440.0                   461.0                         82.0                    61
           Silver Queen    Duvet Cover Set   Light Grey                   1.0          224.0  50.0 178.0  625.0          0.1              395.0                   414.0                         20.0                    61
            Taupe Queen 400 TC Duvet Cover        Taupe                   1.0          210.0  47.0 166.0  585.0          0.1              369.0                   387.0                         82.0                    61
            Taupe  King 400 TC Duvet Cover        Taupe                   1.0          189.0  42.0 150.0  527.0          0.1              332.0                   348.0                         88.0                    61
    Antique White Queen    Duvet Cover Set   Near White                   3.0          177.0  39.0 140.0  493.0          0.2              314.0                   330.0                        125.0                    61
           Silver  King    Duvet Cover Set   Light Grey                   1.0          177.0  39.0 141.0  495.0          0.1              314.0                   329.0                         31.0                    61
            Taupe  King    Duvet Cover Set        Taupe                   0.0          159.0  35.0 126.0  443.0          0.1              279.0                   293.0                         26.0                    61
            Taupe Queen    Duvet Cover Set        Taupe                   0.0          147.0  33.0 116.0  410.0          0.1              258.0                   271.0                         27.0                    49
    Antique White  King    Duvet Cover Set   Near White                   4.0          138.0  31.0 110.0  387.0          0.2              248.0                   260.0                        115.0                    61
```