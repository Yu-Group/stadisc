# StaDISC

This repo is a companion to the paper *Stable discovery of interpretable subgroups via calibration in causal studies* by Raaz Dwivedi, Yan Shuo Tan, Briton Park, Mian Wei, Kevin Horgan, David Madigan, and Bin Yu. We reproduce the paper abstract below.

---

Building on Yu and Kumbier's PCS framework and for randomized experiments, we introduce a novel methodology for Stable Discovery of Interpretable Subgroups via Calibration (StaDISC), with large heterogeneous treatment effects. StaDISC was developed during our re-analysis of the 1999-2000 VIGOR study, an 8076 patient randomized controlled trial, that compared the risk of adverse events from a then newly approved drug, Rofecoxib (Vioxx), to that from an older drug Naproxen. Vioxx was found to, on average and in comparison to Naproxen, reduce the risk of gastrointestinal (GI) events but increase the risk of thrombotic cardiovascular (TC) events. Applying StaDISC, we fit 18 popular conditional average treatment effect (CATE) estimators for both outcomes and use calibration to demonstrate their poor global performance. However, they are locally well-calibrated and stable, enabling the identification of patient groups with larger than (estimated) average treatment effects. In fact, StaDISC discovers three clinically interpretable subgroups for the GI outcome (totaling 29.4% of the study size), and two (totaling 5.6\%) for the TC outcome. Amongst them, the subgroup of people with a prior history of GI events (7.8% of the study size) not only has a disproportionately reduced risk of GI events but also does not experience an increased risk of TC events.

---

In this repo, you will find Jupyter notebooks containing our statistical analysis of the dataset, including the steps we took to generate the results in the paper. Also contained are the scripts used throughout our analysis. The VIGOR data files are confidential and are excluded from the repo. We plan to create a separate package containing tools for applying the StaDISC methodology to othere data sets in the near future.
