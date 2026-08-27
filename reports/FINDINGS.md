# Findings — duvet cover set colour/size cold-start forecasting

Training data through **2026-05-31**; held out **2026-06-01 .. 2026-07-31**.


## 1. Model leaderboard, single held-out window

```
                      cold_wape  cold_mae  cold_bias  cold_spearman  cold_top6_hit  cold_coverage80
poisson_glm               0.605    57.464     -0.361          0.377          0.167            0.789
naive_family              0.608    57.718     -0.167          0.585          0.833            0.684
negbin_glm                0.609    57.855     -0.390          0.222          0.167            0.737
lasso                     0.638    60.567     -0.416          0.187          0.333            0.579
elasticnet                0.659    62.539     -0.412          0.094          0.333            0.579
knn_lookalike             0.660    62.709      0.124          0.451          0.333            0.632
hier_bayes                0.662    62.822      0.145          0.509          0.500            0.895
gompertz                  0.663    62.990     -0.320          0.120          0.333            0.632
logistic                  0.671    63.705     -0.337          0.111          0.167            0.632
bass                      0.675    64.069     -0.343          0.107          0.167            0.579
ensemble                  0.676    64.152      0.013          0.373          0.333            0.737
random_forest             0.676    64.197     -0.001          0.275          0.500            0.789
heuristic                 0.725    68.810     -0.683          0.388          0.500            0.842
ridge                     0.747    70.920     -0.361         -0.203          0.167            0.421
matrix_factorisation      0.753    71.449     -0.254          0.097          0.167            0.579
cluster_pool              0.772    73.331     -0.136          0.251          0.333            0.842
xgboost                   0.775    73.598      0.022          0.012          0.333            0.474
lightgbm                  0.787    74.687      0.354          0.373          0.500            0.421
size_ratio                0.820    77.848      0.125          0.518          0.667            0.842
```

## 2. Rolling-origin validation across 7 launch events

```
                      origins  wape_median  wape_mean  wape_iqr  mean_rank  worst_rank  spearman_median  top6_hit  coverage80  bias_median
model                                                                                                                                     
knn_lookalike               7        0.489      0.818     0.354      4.286         8.0            0.471     0.667       0.595        0.041
negbin_glm                  7        0.636      0.966     0.052      6.286        11.0            0.114     0.667       0.408       -0.416
poisson_glm                 7        0.645      0.994     0.056      6.714        13.0           -0.017     0.643       0.496       -0.394
hier_bayes                  7        0.699      1.215     0.161      7.571        17.0            0.500     0.667       0.649       -0.041
matrix_factorisation        7        0.735      0.910     0.161      7.714        12.0            0.399     0.643       0.356       -0.478
random_forest               7        0.729      0.921     0.310      8.286        15.0            0.248     0.619       0.644       -0.164
heuristic                   7        0.766      0.818     0.088      8.857        15.0            0.388     0.667       0.622       -0.661
lasso                       7        0.683      1.116     0.466      9.000        12.0            0.344     0.643       0.251       -0.533
elasticnet                  7        0.695      1.094     0.505      9.143        13.0            0.400     0.643       0.263       -0.407
ensemble                    7        0.658      1.192     0.258      9.286        16.0            0.400     0.643       0.504       -0.491
cluster_pool                7        0.750      1.002     0.208     10.286        17.0            0.060     0.643       0.541       -0.486
xgboost                     7        0.771      1.122     0.328     11.000        18.0            0.012     0.643       0.479       -0.523
lightgbm                    7        0.750      1.290     0.280     11.714        17.0            0.193     0.619       0.341       -0.537
gompertz                    7        0.863      1.202     0.663     12.000        19.0            0.500     0.667       0.224       -0.300
logistic                    7        0.855      1.258     0.617     12.429        18.0            0.441     0.667       0.233       -0.336
bass                        7        0.852      1.283     0.607     12.429        17.0            0.441     0.667       0.233       -0.344
size_ratio                  7        0.840      2.899     3.852     13.000        19.0            0.500     0.690       0.712        0.408
ridge                       7        0.839      1.668     0.774     13.714        18.0            0.400     0.643       0.177       -0.217
naive_family                7        1.321      3.755     4.470     16.286        19.0            0.499     0.714       0.641        0.881
```

## 3. Order fractile sensitivity vs realised demand

Actual 120-day-equivalent cohort demand: **4,435 units**. Cost index uses stockout:holding = 1.0:0.25.

```
 fractile  total_order  vs_actual  units_short  units_excess  cost_index
      0.4         3535       0.80         1786           886        2007
      0.5         4347       0.98         1422          1335        1756
      0.6         5346       1.21         1070          1981        1566
      0.7         6669       1.50          799          3033        1557
      0.8         8639       1.95          430          4633        1588
      0.9        12369       2.79            0          7934        1983
```

- perfect-hindsight cost index: 0
- incumbent heuristic (2,180 units, flat split): 2,557

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