# Learning Condition Invariant Features for Retrieval-Based Localization from 1M Images

This repository contains the code for our papers  [Learning Condition Invariant Features for Retrieval-Based Localization from 1M Images](https://arxiv.org/pdf/2008.12165.pdf) and [Geometrically Mappable Image Features](https://arxiv.org/abs/2003.09682). 
The corresponding models and training/testing image lists can be downloaded [here](https://www.dropbox.com/sh/xao2zjlp9tbkb1x/AABdGmJUvBcos0pU3JKJYlZVa?dl=0).

This code was tested using TensorFlow 1.10.0 and Python 3.5.6.

It uses the following git repositories as dependencies:

- [netvlad_tf_open](https://github.com/uzh-rpg/netvlad_tf_open)
- [pointnetvlad](https://github.com/mikacuy/pointnetvlad)
- [robotcar-dataset-sdk](https://github.com/ori-mrg/robotcar-dataset-sdk)

The training data can be downloaded using: 

- [RobotCarDataset-Scraper](https://github.com/mttgdd/RobotCarDataset-Scraper)

#### Models used in *Learning Condition Invariant Features for Retrieval-BasedLocalization from 1M Images* Table 2
| Name | Model |
|-|-|
| I: Triplet | triplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| I: Quadruplet | quadruplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| I: Lazy triplet | lazy_triplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| I: Lazy quadruplet | lazy_quadruplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| I: Triplet + distance | sum_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| I: Triplet + Huber dist. | h_sum_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| II: Triplet | triplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| II: Quadruplet | quadruplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| II: Lazy triplet | lazy_triplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| II: Lazy quadruplet | lazy_quadruplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| II: Triplet + Huber dist.~ | h_sum_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| III: Triplet | triplet_xy_000 |
| III: Quadruplet | quadruplet_xy_000 |
| III: Lazy triplet | lazy_triplet_xy_000 |
| III: Lazy quadruplet | lazy_quadruplet_xy_000 |
| III: Triplet + distance | distance_triplet_xy_000 |
| III: Triplet + Huber dist. | huber_distance_triplet_xy_000 |
| **III: Triplet + HP** | evil_triplet_hp_001 |
| **III: Quadruplet + HP** | evil_quadruplet_hp_001 |
| **III: Volume without HP** | ha0_loresidual_det_muTrue_renone_vl64_pca_eccv_000 |
| **III: Volume* without HP** | ha0_loresidual_det_muTrue_renone_vl0_pca_eccv_001 |
| **III: Volume** | residual_det_0.0_eccv_000 |
| **III: Volume*** | residual_det_512_none_0_fc_eccv_000 |
| IV: Triplet | pittsnetvlad |
| V: Off-the-shelf | offtheshelf |

#### Models used in *Geometrically Mappable Image Features* Figure 3
|Name| Oxford Model|
|-|-|
| Off-the-shelf | offtheshelf |
| **Triplet + dist.** | distance_triplet_xy_000 |
| **Triplet + Huber dist.** | huber_distance_triplet_xy_000 |
| **Lazy quad. + dist.** | distance_lazy_quadruplet_xy_000 |
| **Lazy quad. + Huber dist.** | huber_distance_lazy_quadruplet_xy_000 |
| Dist. | distance_xy_000 |
| Huber | huber_distance_xy_000 |
| Triplet | triplet_xy_000 |
| Quadruplet | quadruplet_xy_000 |
| Lazy triplet | lazy_triplet_xy_000 |
| Lazy quadruplet | lazy_quadruplet_xy_000 |

|Name| COLD Model
|-|-|
| Off-the-shelf | offtheshelf |
| **Triplet + dist.** | sum_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| **Triplet + Huber dist.** | h_sum_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| **Lazy quad. + dist.** | nlq_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| **Lazy quad. + Huber dist.** | hlq_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| Dist. | simple_dist_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| Huber | huber_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| Triplet | triplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| Quadruplet | quadruplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| Lazy triplet | lazy_triplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
| Lazy quadruplet | lazy_quadruplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0 |
