# Asset-level syst_share Tracking Report

## Overview

This report bridges two parallel evidence streams:

1. **Weight-shift evidence** (`crisis_weight_test.py`): which assets gained or lost GMV weight during each crisis.

2. **Variance-decomposition evidence** (`variance_decomp.py`): the aggregate cross-sectional relationship between syst_share and GMV weight.


For each (crisis × estimator) cell the **top-5 beneficiaries** and **top-5 casualties** (by Cohen's d on weight) are identified, and their `syst_share = β²σ²_m / total_var` is tracked with a rolling 252-day window (equal-weighted market proxy, stepped every 5 trading days).


---

## Marquee Asset Patterns

JNJ, LMT, and VZ appear across multiple crises as high-signal movers:


### VZ


crisis estimator        side  cohens_d_weight  weight_delta  pre_syst_share  crisis_syst_share  syst_share_delta
   GFC        LW    casualty          -1.6435       -0.0163          0.2901             0.4698            0.1797
   GFC    Gerber    casualty          -1.4794       -0.0259          0.2901             0.4698            0.1797
 COVID    Sample beneficiary           1.8530        0.2470          0.1673             0.3414            0.1740
 COVID        LW beneficiary           1.9759        0.1668          0.1673             0.3414            0.1740
 COVID    Gerber beneficiary           1.5010        0.1646          0.1673             0.3414            0.1740
 Rates    Sample    casualty          -0.8552       -0.1247          0.3434             0.1770           -0.1663


### JNJ


crisis estimator        side  cohens_d_weight  weight_delta  pre_syst_share  crisis_syst_share  syst_share_delta
   GFC    Sample beneficiary           1.6217        0.1581          0.1722             0.3566            0.1845
   GFC        LW beneficiary           1.8226        0.1179          0.1722             0.3566            0.1845
   GFC    Gerber beneficiary           1.3699        0.1672          0.1722             0.3566            0.1845
 Rates    Sample beneficiary           1.1729        0.0583          0.3855             0.1969           -0.1886
 Rates        LW beneficiary           1.2594        0.0516          0.3855             0.1969           -0.1886
 Rates    Gerber beneficiary           1.4253        0.0906          0.3855             0.1969           -0.1886


### LMT


crisis estimator        side  cohens_d_weight  weight_delta  pre_syst_share  crisis_syst_share  syst_share_delta
   GFC    Sample    casualty          -1.4781       -0.0242          0.2106             0.3201            0.1095
   GFC    Gerber    casualty          -1.2722       -0.0272          0.2106             0.3201            0.1095
 Rates    Sample beneficiary           2.5318        0.0652          0.4518             0.1186           -0.3332
 Rates        LW beneficiary           2.8392        0.0654          0.4518             0.1186           -0.3332
 Rates    Gerber beneficiary           1.3666        0.0402          0.4518             0.1186           -0.3332


---

## Full Results by Crisis


### GFC


estimator        side ticker  cohens_d_weight  weight_delta  pre_syst_share  crisis_syst_share  syst_share_delta
   Sample beneficiary     MO           2.0570        0.0396          0.2106             0.2927            0.0820
   Sample beneficiary    JNJ           1.6217        0.1581          0.1722             0.3566            0.1845
   Sample beneficiary     CL           1.2426        0.0388          0.1392             0.3109            0.1717
   Sample beneficiary     KO           0.9932        0.0293          0.3219             0.3647            0.0428
   Sample beneficiary    UNH           0.9604        0.0093          0.1565             0.2033            0.0468
   Sample    casualty   MDLZ          -2.5684       -0.0652          0.1248             0.3201            0.1953
   Sample    casualty      T          -1.8888       -0.0350          0.2737             0.4723            0.1986
   Sample    casualty   MSFT          -1.6140       -0.0211          0.2640             0.4343            0.1703
   Sample    casualty    LMT          -1.4781       -0.0242          0.2106             0.3201            0.1095
   Sample    casualty    CVX          -1.4742       -0.0147          0.2096             0.4403            0.2308
       LW beneficiary     MO           2.3837        0.0462          0.2106             0.2927            0.0820
       LW beneficiary    JNJ           1.8226        0.1179          0.1722             0.3566            0.1845
       LW beneficiary     CL           1.3137        0.0386          0.1392             0.3109            0.1717
       LW beneficiary    UNH           1.0171        0.0109          0.1565             0.2033            0.0468
       LW beneficiary     KO           0.9894        0.0259          0.3219             0.3647            0.0428
       LW    casualty      T          -2.9521       -0.0338          0.2737             0.4723            0.1986
       LW    casualty   MDLZ          -2.2258       -0.0501          0.1248             0.3201            0.1953
       LW    casualty   MSFT          -2.0402       -0.0235          0.2640             0.4343            0.1703
       LW    casualty     VZ          -1.6435       -0.0163          0.2901             0.4698            0.1797
       LW    casualty    CVX          -1.6026       -0.0152          0.2096             0.4403            0.2308
   Gerber beneficiary    JNJ           1.3699        0.1672          0.1722             0.3566            0.1845
   Gerber beneficiary     CL           1.2878        0.0499          0.1392             0.3109            0.1717
   Gerber beneficiary     MO           1.1294        0.0310          0.2106             0.2927            0.0820
   Gerber beneficiary    UNH           0.9200        0.0037          0.1565             0.2033            0.0468
   Gerber beneficiary    XOM           0.8883        0.0046          0.2881             0.4645            0.1764
   Gerber    casualty   MDLZ          -2.4443       -0.0941          0.1248             0.3201            0.1953
   Gerber    casualty     VZ          -1.4794       -0.0259          0.2901             0.4698            0.1797
   Gerber    casualty    LMT          -1.2722       -0.0272          0.2106             0.3201            0.1095
   Gerber    casualty    WMT          -1.2154       -0.0425          0.2588             0.4275            0.1687
   Gerber    casualty   MSFT          -1.1928       -0.0292          0.2640             0.4343            0.1703


### COVID


estimator        side ticker  cohens_d_weight  weight_delta  pre_syst_share  crisis_syst_share  syst_share_delta
   Sample beneficiary    WMT           2.3877        0.0636          0.2168             0.3021            0.0853
   Sample beneficiary     VZ           1.8530        0.2470          0.1673             0.3414            0.1740
   Sample beneficiary   CHTR           1.5111        0.0450          0.1957             0.4134            0.2178
   Sample beneficiary   AMZN           1.4467        0.0748          0.3925             0.4203            0.0278
   Sample beneficiary    BMY           1.3996        0.1095          0.2135             0.3812            0.1677
   Sample    casualty    SLB          -2.9960       -0.0311          0.2570             0.4661            0.2091
   Sample    casualty    DUK          -1.8895       -0.1205          0.0208             0.3890            0.3682
   Sample    casualty    TGT          -1.8150       -0.0127          0.1717             0.2394            0.0677
   Sample    casualty    AIG          -1.8089       -0.0400          0.2683             0.5398            0.2714
   Sample    casualty   BKNG          -1.7442       -0.0322          0.2632             0.5086            0.2454
       LW beneficiary    WMT           2.4925        0.0761          0.2168             0.3021            0.0853
       LW beneficiary   CHTR           2.0961        0.0582          0.1957             0.4134            0.2178
       LW beneficiary     VZ           1.9759        0.1668          0.1673             0.3414            0.1740
       LW beneficiary   COST           1.5522        0.0228          0.3325             0.4109            0.0784
       LW beneficiary   AMZN           1.4686        0.0674          0.3925             0.4203            0.0278
       LW    casualty    SLB          -3.6099       -0.0303          0.2570             0.4661            0.2091
       LW    casualty    DUK          -2.0402       -0.0856          0.0208             0.3890            0.3682
       LW    casualty    AIG          -1.9268       -0.0374          0.2683             0.5398            0.2714
       LW    casualty    TGT          -1.8451       -0.0116          0.1717             0.2394            0.0677
       LW    casualty   SBUX          -1.8389       -0.0262          0.2126             0.5341            0.3215
   Gerber beneficiary    WMT           2.3575        0.0664          0.2168             0.3021            0.0853
   Gerber beneficiary   CHTR           1.7649        0.0538          0.1957             0.4134            0.2178
   Gerber beneficiary     VZ           1.5010        0.1646          0.1673             0.3414            0.1740
   Gerber beneficiary   AMZN           1.3008        0.0751          0.3925             0.4203            0.0278
   Gerber beneficiary     MO           1.1966        0.0420          0.1332             0.3267            0.1935
   Gerber    casualty     KO          -1.6812       -0.0941          0.2329             0.4240            0.1911
   Gerber    casualty    AIG          -1.4670       -0.0486          0.2683             0.5398            0.2714
   Gerber    casualty    SLB          -1.1824       -0.0272          0.2570             0.4661            0.2091
   Gerber    casualty    DUK          -1.1584       -0.1163          0.0208             0.3890            0.3682
   Gerber    casualty    BAC          -1.0943       -0.0290          0.4801             0.7209            0.2408


### Rates


estimator        side ticker  cohens_d_weight  weight_delta  pre_syst_share  crisis_syst_share  syst_share_delta
   Sample beneficiary    MRK           3.0832        0.0422          0.3932             0.0919           -0.3013
   Sample beneficiary    LMT           2.5318        0.0652          0.4518             0.1186           -0.3332
   Sample beneficiary    IBM           1.3322        0.0121          0.6011             0.2543           -0.3468
   Sample beneficiary    JNJ           1.1729        0.0583          0.3855             0.1969           -0.1886
   Sample beneficiary    PFE           1.1381        0.0228          0.3720             0.0860           -0.2860
   Sample    casualty      D          -1.0522       -0.0191          0.2970             0.1427           -0.1543
   Sample    casualty   QCOM          -0.8624       -0.0076          0.3746             0.3408           -0.0338
   Sample    casualty     VZ          -0.8552       -0.1247          0.3434             0.1770           -0.1663
   Sample    casualty    WFC          -0.8385       -0.0115          0.5939             0.4609           -0.1329
   Sample    casualty   AMZN          -0.8097       -0.0386          0.3581             0.2891           -0.0690
       LW beneficiary    MRK           3.5570        0.0436          0.3932             0.0919           -0.3013
       LW beneficiary    LMT           2.8392        0.0654          0.4518             0.1186           -0.3332
       LW beneficiary    IBM           1.5072        0.0146          0.6011             0.2543           -0.3468
       LW beneficiary     PM           1.3023        0.0112          0.4110             0.2009           -0.2101
       LW beneficiary    JNJ           1.2594        0.0516          0.3855             0.1969           -0.1886
       LW    casualty      D          -1.1288       -0.0228          0.2970             0.1427           -0.1543
       LW    casualty   CHTR          -1.0263       -0.0289          0.3816             0.1610           -0.2206
       LW    casualty   COST          -0.9736       -0.0290          0.3664             0.2753           -0.0911
       LW    casualty   QCOM          -0.8882       -0.0079          0.3746             0.3408           -0.0338
       LW    casualty    WFC          -0.8568       -0.0108          0.5939             0.4609           -0.1329
   Gerber beneficiary    MCD           1.4844        0.0668          0.4344             0.3519           -0.0825
   Gerber beneficiary    JNJ           1.4253        0.0906          0.3855             0.1969           -0.1886
   Gerber beneficiary    LMT           1.3666        0.0402          0.4518             0.1186           -0.3332
   Gerber beneficiary    CVX           1.2444        0.0261          0.5661             0.3022           -0.2639
   Gerber beneficiary     PG           1.1177        0.0652          0.3561             0.1975           -0.1587
   Gerber    casualty    WMT          -1.0256       -0.0336          0.2590             0.1406           -0.1184
   Gerber    casualty    AMT          -1.0063       -0.0385          0.3199             0.2019           -0.1180
   Gerber    casualty    WFC          -0.9512       -0.0241          0.5939             0.4609           -0.1329
   Gerber    casualty     MO          -0.9387       -0.0276          0.3522             0.2055           -0.1467
   Gerber    casualty   AMZN          -0.7598       -0.0410          0.3581             0.2891           -0.0690

