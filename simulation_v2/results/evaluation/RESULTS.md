# Simulation v2 of RabMV: results (evaluation level)

Synthetic data only. Pre-specified criteria (simulation_v2/PLAN.md):

- **C1_transport**: met. E_pi100, target T3: upper 95 % limit of mean Δ excess log-loss (RabMV − comparator) < 0 against each comparator.
- **C2_no_cost_of_the_rule**: met. MAR (blanks made by people), T0: 90 % CI of Δ AUC-PR within ±0.01 for the rule against the same engine without it.
- **C3_detection**: met. Wilson lower limit of the sensitivity (E_pi100, undeclared schema) ≥ 0.9 and of the per-concept specificity (no_difference) ≥ 0.99.
- **C4_localization**: NOT met. Wilson lower limit of P(gap concept alerted and no honest concept alerted) ≥ 0.9 (E_pi100, undeclared schema).

Method validity (all criteria): **False**

## Excess log-loss over the oracle, 180 events (mean, MCSE)

### E_pi100_education

| method            |      T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|--------:|-------:|-------:|-------:|-------:|
| rabmv             |  0.0247 | 0.0247 | 0.0247 | 0.0249 | 0.0247 |
| rabmv_impute      |  0.0249 | 0.025  | 0.025  | 0.0261 | 0.0249 |
| rabmv_indicator   |  0.0151 | 0.0251 | 0.0357 | 0.2456 | 0.0151 |
| rabmv_mice        |  0.0248 | 0.0249 | 0.0249 | 0.0256 | 0.0248 |
| rabmv_pattern     |  0.0242 | 0.0255 | 0.0255 | 0.0345 | 0.0242 |
| drop              |  0.025  | 0.025  | 0.025  | 0.025  | 0.025  |
| ebm_native        |  0.0148 | 0.0248 | 0.0361 | 0.2579 | 0.0148 |
| ebm_marginalize   |  0.0247 | 0.0248 | 0.0248 | 0.0249 | 0.0247 |
| ebm_impute        |  0.024  | 0.0244 | 0.0244 | 0.0272 | 0.024  |
| ebm_indicator     |  0.0143 | 0.0246 | 0.0397 | 0.312  | 0.0143 |
| ebm_mice          |  0.024  | 0.0241 | 0.0242 | 0.0255 | 0.024  |
| ebm_pattern       |  0.0233 | 0.0246 | 0.0246 | 0.0335 | 0.0233 |
| trees_native      |  0.0697 | 0.0799 | 0.1063 | 0.5226 | 0.0697 |
| rabmv_sensitivity |  0.0129 | 0.0563 | 0.0423 | 0.0131 | 0.0314 |
| naive             | -0.0292 | 0.2438 | 0.1575 | 0.0979 | 0.0842 |
| oracle            |  0      | 0      | 0      | 0      | 0      |

### E_pi85_education

| method            |      T0 |     T1 |     T2 |      T3 |     T4 |
|:------------------|--------:|-------:|-------:|--------:|-------:|
| rabmv             |  0.0266 | 0.024  | 0.0247 |  0.0267 | 0.0255 |
| rabmv_impute      |  0.0147 | 0.059  | 0.0436 |  0.0151 | 0.0336 |
| rabmv_indicator   | -0.0413 | 0.0863 | 0.1168 | -0.0359 | 0.0134 |
| rabmv_mice        |  0.0241 | 0.023  | 0.024  |  0.0245 | 0.0236 |
| rabmv_pattern     | -0.0034 | 0.1087 | 0.0666 |  0.0011 | 0.0445 |
| drop              |  0.0268 | 0.0242 | 0.0249 |  0.0268 | 0.0257 |
| ebm_native        | -0.0416 | 0.0823 | 0.1119 | -0.0374 | 0.0114 |
| ebm_marginalize   |  0.0826 | 0.0823 | 0.0824 |  0.0829 | 0.0825 |
| ebm_impute        |  0.0142 | 0.0636 | 0.0458 |  0.0151 | 0.0352 |
| ebm_indicator     | -0.0416 | 0.0852 | 0.1157 | -0.0323 | 0.0128 |
| ebm_mice          |  0.0232 | 0.0222 | 0.0232 |  0.024  | 0.0228 |
| ebm_pattern       | -0.0035 | 0.1118 | 0.0675 |  0.001  | 0.0458 |
| trees_native      |  0.0094 | 0.2507 | 0.2327 |  0.0231 | 0.1129 |
| rabmv_sensitivity |  0.0291 | 0.0125 | 0.0166 |  0.0293 | 0.0221 |
| naive             |  0.0064 | 0.062  | 0.0468 |  0.0407 | 0.029  |
| oracle            |  0      | 0      | 0      |  0      | 0      |

### E_pi50_education

| method            |     T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|-------:|-------:|-------:|-------:|-------:|
| rabmv             | 0.0237 | 0.0226 | 0.0237 | 0.0237 | 0.0232 |
| rabmv_impute      | 0.0241 | 0.023  | 0.0241 | 0.0241 | 0.0237 |
| rabmv_indicator   | 0.024  | 0.0229 | 0.024  | 0.0252 | 0.0236 |
| rabmv_mice        | 0.0241 | 0.0231 | 0.024  | 0.0241 | 0.0236 |
| rabmv_pattern     | 0.0264 | 0.0279 | 0.0264 | 0.0264 | 0.027  |
| drop              | 0.0238 | 0.0227 | 0.0238 | 0.0238 | 0.0233 |
| ebm_native        | 0.0232 | 0.022  | 0.0232 | 0.0238 | 0.0227 |
| ebm_marginalize   | 0.0231 | 0.022  | 0.0231 | 0.0232 | 0.0227 |
| ebm_impute        | 0.0234 | 0.0222 | 0.0234 | 0.0233 | 0.0229 |
| ebm_indicator     | 0.0233 | 0.0222 | 0.0233 | 0.0246 | 0.0229 |
| ebm_mice          | 0.0233 | 0.0222 | 0.0232 | 0.0233 | 0.0228 |
| ebm_pattern       | 0.0253 | 0.0269 | 0.0253 | 0.0254 | 0.026  |
| trees_native      | 0.0808 | 0.0803 | 0.0808 | 0.0836 | 0.0809 |
| rabmv_sensitivity | 0.0141 | 0.0089 | 0.0141 | 0.0142 | 0.012  |
| naive             | 0.0149 | 0.0103 | 0.0149 | 0.0157 | 0.013  |
| oracle            | 0      | 0      | 0      | 0      | 0      |

### MAR_education

| method            |     T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|-------:|-------:|-------:|-------:|-------:|
| rabmv             | 0.0237 | 0.0226 | 0.0237 | 0.0238 | 0.0233 |
| rabmv_impute      | 0.0241 | 0.023  | 0.0241 | 0.0241 | 0.0236 |
| rabmv_indicator   | 0.0241 | 0.0231 | 0.0241 | 0.0246 | 0.0237 |
| rabmv_mice        | 0.024  | 0.023  | 0.024  | 0.0241 | 0.0236 |
| rabmv_pattern     | 0.0263 | 0.0273 | 0.0263 | 0.0261 | 0.0268 |
| drop              | 0.0238 | 0.0227 | 0.0238 | 0.0238 | 0.0233 |
| ebm_native        | 0.0233 | 0.0222 | 0.0233 | 0.0237 | 0.0228 |
| ebm_marginalize   | 0.0232 | 0.022  | 0.0232 | 0.0232 | 0.0227 |
| ebm_impute        | 0.0234 | 0.0223 | 0.0234 | 0.0233 | 0.0229 |
| ebm_indicator     | 0.0235 | 0.0225 | 0.0235 | 0.0241 | 0.0231 |
| ebm_mice          | 0.0233 | 0.0222 | 0.0232 | 0.0233 | 0.0228 |
| ebm_pattern       | 0.0255 | 0.0265 | 0.0255 | 0.0252 | 0.026  |
| trees_native      | 0.0807 | 0.0809 | 0.0807 | 0.0822 | 0.0809 |
| rabmv_sensitivity | 0.0142 | 0.0089 | 0.0142 | 0.0142 | 0.012  |
| naive             | 0.0151 | 0.0104 | 0.0151 | 0.0149 | 0.0131 |
| oracle            | 0      | 0      | 0      | 0      | 0      |

### MNAR_education

| method            |     T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|-------:|-------:|-------:|-------:|-------:|
| rabmv             | 0.0237 | 0.0226 | 0.0237 | 0.0237 | 0.0232 |
| rabmv_impute      | 0.0241 | 0.023  | 0.0241 | 0.0241 | 0.0236 |
| rabmv_indicator   | 0.0241 | 0.023  | 0.0241 | 0.0246 | 0.0236 |
| rabmv_mice        | 0.024  | 0.023  | 0.024  | 0.0241 | 0.0236 |
| rabmv_pattern     | 0.0264 | 0.0275 | 0.0264 | 0.0262 | 0.0269 |
| drop              | 0.0238 | 0.0227 | 0.0238 | 0.0238 | 0.0233 |
| ebm_native        | 0.0232 | 0.0221 | 0.0232 | 0.0236 | 0.0227 |
| ebm_marginalize   | 0.023  | 0.0219 | 0.023  | 0.0231 | 0.0226 |
| ebm_impute        | 0.0234 | 0.0223 | 0.0234 | 0.0233 | 0.0229 |
| ebm_indicator     | 0.0233 | 0.0223 | 0.0233 | 0.0238 | 0.0229 |
| ebm_mice          | 0.0232 | 0.0222 | 0.0232 | 0.0233 | 0.0228 |
| ebm_pattern       | 0.0254 | 0.0264 | 0.0254 | 0.0252 | 0.0259 |
| trees_native      | 0.0812 | 0.0812 | 0.0812 | 0.0828 | 0.0812 |
| rabmv_sensitivity | 0.0141 | 0.0089 | 0.0141 | 0.0142 | 0.012  |
| naive             | 0.015  | 0.0103 | 0.015  | 0.015  | 0.013  |
| oracle            | 0      | 0      | 0      | 0      | 0      |

### E_pi100_occupation

| method            |      T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|--------:|-------:|-------:|-------:|-------:|
| rabmv             |  0.0245 | 0.0247 | 0.0249 | 0.0291 | 0.0245 |
| rabmv_impute      |  0.0228 | 0.0249 | 0.0253 | 0.0454 | 0.0228 |
| rabmv_indicator   |  0.0151 | 0.0251 | 0.0363 | 0.2543 | 0.0151 |
| rabmv_mice        |  0.0253 | 0.025  | 0.0252 | 0.0268 | 0.0252 |
| rabmv_pattern     |  0.0243 | 0.0255 | 0.0257 | 0.0371 | 0.0243 |
| drop              |  0.0281 | 0.0281 | 0.0281 | 0.0281 | 0.0281 |
| ebm_native        |  0.0145 | 0.0245 | 0.0359 | 0.2576 | 0.0145 |
| ebm_marginalize   |  0.0244 | 0.0245 | 0.0248 | 0.0287 | 0.0244 |
| ebm_impute        |  0.0221 | 0.0242 | 0.0246 | 0.0452 | 0.0221 |
| ebm_indicator     |  0.0142 | 0.0246 | 0.0407 | 0.3271 | 0.0142 |
| ebm_mice          |  0.0245 | 0.0242 | 0.0245 | 0.0261 | 0.0245 |
| ebm_pattern       |  0.0234 | 0.0246 | 0.0248 | 0.0362 | 0.0234 |
| trees_native      |  0.0696 | 0.0799 | 0.1049 | 0.5049 | 0.0696 |
| rabmv_sensitivity |  0.0128 | 0.0563 | 0.0426 | 0.0172 | 0.0313 |
| naive             | -0.0257 | 0.2383 | 0.1484 | 0.0153 | 0.0855 |
| oracle            |  0      | 0      | 0      | 0      | 0      |

### E_pi85_occupation

| method            |      T0 |     T1 |     T2 |      T3 |     T4 |
|:------------------|--------:|-------:|-------:|--------:|-------:|
| rabmv             |  0.0265 | 0.024  | 0.0249 |  0.0301 | 0.0255 |
| rabmv_impute      |  0.0136 | 0.0585 | 0.044  |  0.0269 | 0.0323 |
| rabmv_indicator   | -0.0412 | 0.0859 | 0.1171 | -0.0332 | 0.0133 |
| rabmv_mice        |  0.0244 | 0.0231 | 0.0243 |  0.026  | 0.024  |
| rabmv_pattern     | -0.0033 | 0.1087 | 0.0668 |  0.0034 | 0.0445 |
| drop              |  0.0298 | 0.0272 | 0.0279 |  0.0298 | 0.0288 |
| ebm_native        | -0.0414 | 0.0816 | 0.1128 | -0.0312 | 0.0113 |
| ebm_marginalize   |  0.0821 | 0.0816 | 0.082  |  0.0859 | 0.0819 |
| ebm_impute        |  0.0131 | 0.0629 | 0.0461 |  0.0266 | 0.0339 |
| ebm_indicator     | -0.0415 | 0.0849 | 0.116  | -0.0296 | 0.0128 |
| ebm_mice          |  0.0236 | 0.0223 | 0.0236 |  0.0252 | 0.0232 |
| ebm_pattern       | -0.0034 | 0.1118 | 0.0678 |  0.0034 | 0.0459 |
| trees_native      |  0.0093 | 0.248  | 0.2313 |  0.0141 | 0.1117 |
| rabmv_sensitivity |  0.0291 | 0.0125 | 0.0168 |  0.0329 | 0.022  |
| naive             |  0.0091 | 0.0631 | 0.0466 |  0.02   | 0.0316 |
| oracle            |  0      | 0      | 0      |  0      | 0      |

### E_pi50_occupation

| method            |     T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|-------:|-------:|-------:|-------:|-------:|
| rabmv             | 0.0239 | 0.0226 | 0.0239 | 0.0252 | 0.0235 |
| rabmv_impute      | 0.0243 | 0.023  | 0.0243 | 0.0269 | 0.0239 |
| rabmv_indicator   | 0.0243 | 0.0229 | 0.0243 | 0.0268 | 0.0239 |
| rabmv_mice        | 0.0244 | 0.0231 | 0.0243 | 0.0259 | 0.0239 |
| rabmv_pattern     | 0.0266 | 0.0279 | 0.0266 | 0.0279 | 0.0272 |
| drop              | 0.0269 | 0.0259 | 0.0269 | 0.0269 | 0.0265 |
| ebm_native        | 0.0236 | 0.0222 | 0.0236 | 0.026  | 0.0232 |
| ebm_marginalize   | 0.0235 | 0.0222 | 0.0235 | 0.0248 | 0.0231 |
| ebm_impute        | 0.0236 | 0.0223 | 0.0236 | 0.0263 | 0.0232 |
| ebm_indicator     | 0.0236 | 0.0222 | 0.0236 | 0.0262 | 0.0231 |
| ebm_mice          | 0.0236 | 0.0223 | 0.0236 | 0.0252 | 0.0232 |
| ebm_pattern       | 0.0256 | 0.0269 | 0.0256 | 0.027  | 0.0262 |
| trees_native      | 0.0804 | 0.0799 | 0.0804 | 0.0835 | 0.0805 |
| rabmv_sensitivity | 0.0144 | 0.009  | 0.0144 | 0.016  | 0.0122 |
| naive             | 0.0169 | 0.0123 | 0.0169 | 0.0179 | 0.015  |
| oracle            | 0      | 0      | 0      | 0      | 0      |

### MAR_occupation

| method            |     T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|-------:|-------:|-------:|-------:|-------:|
| rabmv             | 0.0242 | 0.0231 | 0.0242 | 0.0255 | 0.0237 |
| rabmv_impute      | 0.0247 | 0.0237 | 0.0247 | 0.027  | 0.0243 |
| rabmv_indicator   | 0.0245 | 0.0235 | 0.0245 | 0.0263 | 0.0241 |
| rabmv_mice        | 0.0247 | 0.0236 | 0.0246 | 0.0262 | 0.0242 |
| rabmv_pattern     | 0.0267 | 0.0277 | 0.0267 | 0.0278 | 0.0272 |
| drop              | 0.0269 | 0.0259 | 0.0269 | 0.0269 | 0.0265 |
| ebm_native        | 0.0239 | 0.0228 | 0.0239 | 0.0255 | 0.0234 |
| ebm_marginalize   | 0.0238 | 0.0227 | 0.0238 | 0.0251 | 0.0233 |
| ebm_impute        | 0.024  | 0.0229 | 0.024  | 0.0263 | 0.0235 |
| ebm_indicator     | 0.0239 | 0.0228 | 0.0239 | 0.0257 | 0.0234 |
| ebm_mice          | 0.0239 | 0.0228 | 0.0239 | 0.0254 | 0.0234 |
| ebm_pattern       | 0.026  | 0.027  | 0.026  | 0.0271 | 0.0265 |
| trees_native      | 0.0805 | 0.0806 | 0.0805 | 0.0827 | 0.0805 |
| rabmv_sensitivity | 0.0147 | 0.0095 | 0.0147 | 0.0162 | 0.0125 |
| naive             | 0.0172 | 0.0126 | 0.0172 | 0.0179 | 0.0153 |
| oracle            | 0      | 0      | 0      | 0      | 0      |

### MNAR_occupation

| method            |     T0 |     T1 |     T2 |     T3 |     T4 |
|:------------------|-------:|-------:|-------:|-------:|-------:|
| rabmv             | 0.0241 | 0.023  | 0.0241 | 0.0255 | 0.0236 |
| rabmv_impute      | 0.0242 | 0.0231 | 0.0242 | 0.0271 | 0.0238 |
| rabmv_indicator   | 0.0242 | 0.0232 | 0.0242 | 0.0268 | 0.0238 |
| rabmv_mice        | 0.0248 | 0.0238 | 0.0248 | 0.0262 | 0.0244 |
| rabmv_pattern     | 0.0267 | 0.0276 | 0.0267 | 0.0278 | 0.0272 |
| drop              | 0.0269 | 0.0259 | 0.0269 | 0.0269 | 0.0265 |
| ebm_native        | 0.0235 | 0.0224 | 0.0235 | 0.0261 | 0.023  |
| ebm_marginalize   | 0.0237 | 0.0226 | 0.0237 | 0.025  | 0.0232 |
| ebm_impute        | 0.0235 | 0.0224 | 0.0235 | 0.0264 | 0.023  |
| ebm_indicator     | 0.0235 | 0.0224 | 0.0235 | 0.0261 | 0.023  |
| ebm_mice          | 0.024  | 0.023  | 0.024  | 0.0254 | 0.0236 |
| ebm_pattern       | 0.0259 | 0.0269 | 0.0259 | 0.027  | 0.0265 |
| trees_native      | 0.0802 | 0.0805 | 0.0802 | 0.0847 | 0.0804 |
| rabmv_sensitivity | 0.0146 | 0.0093 | 0.0146 | 0.0162 | 0.0124 |
| naive             | 0.0169 | 0.0123 | 0.0169 | 0.0181 | 0.015  |
| oracle            | 0      | 0      | 0      | 0      | 0      |

### no_difference

| method            |     T0 |
|:------------------|-------:|
| rabmv             | 0.0072 |
| rabmv_impute      | 0.0072 |
| rabmv_indicator   | 0.0072 |
| rabmv_mice        | 0.0072 |
| rabmv_pattern     | 0.0072 |
| ebm_native        | 0.0064 |
| ebm_marginalize   | 0.0064 |
| ebm_impute        | 0.0064 |
| ebm_indicator     | 0.0064 |
| ebm_mice          | 0.0064 |
| ebm_pattern       | 0.0064 |
| trees_native      | 0.0632 |
| rabmv_sensitivity | 0.0072 |
| naive             | 0.0068 |
| oracle            | 0      |

### no_difference_year

| method            |     T0 |
|:------------------|-------:|
| rabmv             | 0.0037 |
| rabmv_impute      | 0.0037 |
| rabmv_indicator   | 0.0037 |
| rabmv_mice        | 0.0037 |
| rabmv_pattern     | 0.0037 |
| ebm_native        | 0.0029 |
| ebm_marginalize   | 0.0029 |
| ebm_impute        | 0.0029 |
| ebm_indicator     | 0.0029 |
| ebm_mice          | 0.0029 |
| ebm_pattern       | 0.0029 |
| trees_native      | 0.0624 |
| rabmv_sensitivity | 0.0037 |
| naive             | 0.0031 |
| oracle            | 0      |

## Shortcut and shape recovery (T0 and T1)

| condition          |   events | method          |   shortcut_abs_mean |   shortcut_abs_mcse |   shortcut_aucpr_mean |   shortcut_aucpr_mcse |   shape_rmse_mean |   shape_rmse_mcse |
|:-------------------|---------:|:----------------|--------------------:|--------------------:|----------------------:|----------------------:|------------------:|------------------:|
| E_pi100_education  |      180 | rabmv           |              0.0179 |              0.0009 |                0.0001 |                0      |            0.0719 |            0.0029 |
| E_pi100_education  |      180 | rabmv_impute    |              0.0269 |              0.0014 |                0.0009 |                0.0001 |            0.0965 |            0.0029 |
| E_pi100_education  |      180 | rabmv_indicator |              3.1832 |              0.0551 |                0.0306 |                0.0003 |            0.0775 |            0.0028 |
| E_pi100_education  |      180 | rabmv_mice      |              0.0276 |              0.0012 |                0.0005 |                0      |            0.0694 |            0.0028 |
| E_pi100_education  |      180 | rabmv_pattern   |              0.1831 |              0.0072 |                0.0061 |                0.0003 |            0.0734 |            0.0028 |
| E_pi100_education  |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi100_education  |      180 | ebm_native      |              3.3713 |              0.0842 |                0.0297 |                0.0003 |            0.1312 |            0.0048 |
| E_pi100_education  |      180 | ebm_marginalize |              0.0429 |              0.0019 |                0.0002 |                0      |            0.1312 |            0.0048 |
| E_pi100_education  |      180 | ebm_impute      |              0.0684 |              0.0037 |                0.0021 |                0.0002 |            0.129  |            0.0051 |
| E_pi100_education  |      180 | ebm_indicator   |              4.0873 |              0.0325 |                0.0299 |                0.0003 |            0.0978 |            0.0042 |
| E_pi100_education  |      180 | ebm_mice        |              0.0627 |              0.0023 |                0.0009 |                0.0001 |            0.0938 |            0.0041 |
| E_pi100_education  |      180 | ebm_pattern     |              0.1891 |              0.0052 |                0.0058 |                0.0002 |            0.0931 |            0.0041 |
| E_pi100_education  |      180 | trees_native    |              4.5279 |              0.0377 |                0.0264 |                0.0003 |            0.2456 |            0.009  |
| E_pi100_education  |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi100_education  |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi100_occupation |      180 | rabmv           |              0.1243 |              0.0037 |                0.0013 |                0.0001 |            0.0863 |            0.006  |
| E_pi100_occupation |      180 | rabmv_impute    |              0.2977 |              0.0075 |                0.0107 |                0.0002 |            0.0632 |            0.0048 |
| E_pi100_occupation |      180 | rabmv_indicator |              3.2566 |              0.0533 |                0.0301 |                0.0003 |            0.0793 |            0.0056 |
| E_pi100_occupation |      180 | rabmv_mice      |              0.1433 |              0.0043 |               -0.0013 |                0.0001 |            0.1071 |            0.0067 |
| E_pi100_occupation |      180 | rabmv_pattern   |              0.1985 |              0.0054 |                0.0069 |                0.0003 |            0.0972 |            0.0062 |
| E_pi100_occupation |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi100_occupation |      180 | ebm_native      |              3.3249 |              0.0495 |                0.0298 |                0.0003 |            0.0862 |            0.0058 |
| E_pi100_occupation |      180 | ebm_marginalize |              0.1197 |              0.0032 |                0.0011 |                0.0001 |            0.0862 |            0.0058 |
| E_pi100_occupation |      180 | ebm_impute      |              0.3079 |              0.0076 |                0.0104 |                0.0002 |            0.0627 |            0.0052 |
| E_pi100_occupation |      180 | ebm_indicator   |              4.2435 |              0.0299 |                0.0298 |                0.0003 |            0.0841 |            0.0059 |
| E_pi100_occupation |      180 | ebm_mice        |              0.1481 |              0.0045 |               -0.001  |                0.0001 |            0.1012 |            0.0067 |
| E_pi100_occupation |      180 | ebm_pattern     |              0.213  |              0.0044 |                0.0063 |                0.0002 |            0.0925 |            0.0062 |
| E_pi100_occupation |      180 | trees_native    |              4.1105 |              0.0273 |                0.0268 |                0.0003 |            0.123  |            0.0103 |
| E_pi100_occupation |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi100_occupation |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi50_education   |      180 | rabmv           |              0.1079 |              0.004  |                0.0019 |                0.0002 |            0.0747 |            0.0031 |
| E_pi50_education   |      180 | rabmv_impute    |              0.166  |              0.0074 |               -0.0005 |                0.0003 |            0.0763 |            0.003  |
| E_pi50_education   |      180 | rabmv_indicator |              0.182  |              0.0078 |               -0.0005 |                0.0004 |            0.0754 |            0.003  |
| E_pi50_education   |      180 | rabmv_mice      |              0.078  |              0.0025 |               -0.0001 |                0.0002 |            0.0765 |            0.003  |
| E_pi50_education   |      180 | rabmv_pattern   |              0.3458 |              0.008  |                0.0098 |                0.0008 |            0.0778 |            0.0037 |
| E_pi50_education   |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi50_education   |      180 | ebm_native      |              0.1631 |              0.007  |               -0.0005 |                0.0003 |            0.0849 |            0.004  |
| E_pi50_education   |      180 | ebm_marginalize |              0.0887 |              0.0032 |                0.0012 |                0.0002 |            0.0849 |            0.004  |
| E_pi50_education   |      180 | ebm_impute      |              0.1766 |              0.0075 |               -0.0005 |                0.0002 |            0.0988 |            0.0042 |
| E_pi50_education   |      180 | ebm_indicator   |              0.1869 |              0.0073 |               -0.0004 |                0.0004 |            0.0969 |            0.0041 |
| E_pi50_education   |      180 | ebm_mice        |              0.0859 |              0.0025 |               -0.0003 |                0.0001 |            0.0947 |            0.004  |
| E_pi50_education   |      180 | ebm_pattern     |              0.3564 |              0.0071 |                0.0096 |                0.0009 |            0.1432 |            0.007  |
| E_pi50_education   |      180 | trees_native    |              0.7997 |              0.0094 |               -0.0057 |                0.0011 |            0.2178 |            0.0082 |
| E_pi50_education   |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi50_education   |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi50_occupation  |      180 | rabmv           |              0.1199 |              0.0041 |                0.0021 |                0.0002 |            0.0931 |            0.0067 |
| E_pi50_occupation  |      180 | rabmv_impute    |              0.1954 |              0.0077 |               -0.0002 |                0.0003 |            0.0983 |            0.0064 |
| E_pi50_occupation  |      180 | rabmv_indicator |              0.1866 |              0.0079 |               -0.0003 |                0.0004 |            0.0959 |            0.0067 |
| E_pi50_occupation  |      180 | rabmv_mice      |              0.0927 |              0.0025 |               -0.0001 |                0.0002 |            0.1073 |            0.0067 |
| E_pi50_occupation  |      180 | rabmv_pattern   |              0.3475 |              0.0081 |                0.0101 |                0.0008 |            0.139  |            0.0085 |
| E_pi50_occupation  |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi50_occupation  |      180 | ebm_native      |              0.1738 |              0.0075 |               -0.0005 |                0.0003 |            0.1187 |            0.0066 |
| E_pi50_occupation  |      180 | ebm_marginalize |              0.098  |              0.0032 |                0.0013 |                0.0002 |            0.1187 |            0.0066 |
| E_pi50_occupation  |      180 | ebm_impute      |              0.2037 |              0.0077 |               -0.0003 |                0.0003 |            0.0899 |            0.0064 |
| E_pi50_occupation  |      180 | ebm_indicator   |              0.1898 |              0.0073 |               -0.0001 |                0.0004 |            0.0889 |            0.0065 |
| E_pi50_occupation  |      180 | ebm_mice        |              0.0972 |              0.0025 |               -0.0004 |                0.0001 |            0.0996 |            0.0068 |
| E_pi50_occupation  |      180 | ebm_pattern     |              0.36   |              0.0071 |                0.0097 |                0.0009 |            0.1177 |            0.0083 |
| E_pi50_occupation  |      180 | trees_native    |              0.7897 |              0.0088 |               -0.0052 |                0.0011 |            0.1229 |            0.0093 |
| E_pi50_occupation  |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi50_occupation  |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi85_education   |      180 | rabmv           |              0.0802 |              0.0037 |               -0.0033 |                0.0003 |            0.0727 |            0.003  |
| E_pi85_education   |      180 | rabmv_impute    |              0.3031 |              0.0038 |                0.1106 |                0.0015 |            0.0879 |            0.0029 |
| E_pi85_education   |      180 | rabmv_indicator |              2.2988 |              0.0209 |                0.2974 |                0.0017 |            0.0753 |            0.0029 |
| E_pi85_education   |      180 | rabmv_mice      |              0.0589 |              0.0015 |                0.0025 |                0.0002 |            0.0709 |            0.0028 |
| E_pi85_education   |      180 | rabmv_pattern   |              1.2998 |              0.0252 |                0.3353 |                0.0038 |            0.0953 |            0.0055 |
| E_pi85_education   |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi85_education   |      180 | ebm_native      |              2.2328 |              0.0211 |                0.2918 |                0.0016 |            0.0965 |            0.004  |
| E_pi85_education   |      180 | ebm_marginalize |              0.0648 |              0.003  |               -0.004  |                0.0004 |            0.0965 |            0.004  |
| E_pi85_education   |      180 | ebm_impute      |              0.3349 |              0.004  |                0.1125 |                0.0017 |            0.1124 |            0.0047 |
| E_pi85_education   |      180 | ebm_indicator   |              2.3401 |              0.0295 |                0.2921 |                0.0017 |            0.1032 |            0.0043 |
| E_pi85_education   |      180 | ebm_mice        |              0.0731 |              0.0017 |                0.003  |                0.0002 |            0.0926 |            0.0041 |
| E_pi85_education   |      180 | ebm_pattern     |              1.4146 |              0.0314 |                0.33   |                0.004  |            0.2064 |            0.0123 |
| E_pi85_education   |      180 | trees_native    |              3.5756 |              0.0318 |                0.3156 |                0.0034 |            0.2558 |            0.0117 |
| E_pi85_education   |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi85_education   |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi85_occupation  |      180 | rabmv           |              0.1154 |              0.0041 |               -0.0024 |                0.0003 |            0.0969 |            0.0064 |
| E_pi85_occupation  |      180 | rabmv_impute    |              0.3878 |              0.005  |                0.1159 |                0.0015 |            0.0691 |            0.0051 |
| E_pi85_occupation  |      180 | rabmv_indicator |              2.3186 |              0.0213 |                0.2978 |                0.0017 |            0.106  |            0.0067 |
| E_pi85_occupation  |      180 | rabmv_mice      |              0.101  |              0.0022 |                0.0014 |                0.0002 |            0.1067 |            0.0068 |
| E_pi85_occupation  |      180 | rabmv_pattern   |              1.2928 |              0.0251 |                0.3359 |                0.0038 |            0.2031 |            0.0141 |
| E_pi85_occupation  |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi85_occupation  |      180 | ebm_native      |              2.2921 |              0.0246 |                0.2925 |                0.0016 |            0.1233 |            0.0072 |
| E_pi85_occupation  |      180 | ebm_marginalize |              0.0892 |              0.0034 |               -0.0033 |                0.0004 |            0.1233 |            0.0072 |
| E_pi85_occupation  |      180 | ebm_impute      |              0.4104 |              0.0052 |                0.1174 |                0.0017 |            0.0659 |            0.0052 |
| E_pi85_occupation  |      180 | ebm_indicator   |              2.3521 |              0.0308 |                0.2927 |                0.0017 |            0.1005 |            0.0071 |
| E_pi85_occupation  |      180 | ebm_mice        |              0.1052 |              0.0021 |                0.0019 |                0.0002 |            0.0994 |            0.0068 |
| E_pi85_occupation  |      180 | ebm_pattern     |              1.4147 |              0.0314 |                0.3305 |                0.004  |            0.2016 |            0.0146 |
| E_pi85_occupation  |      180 | trees_native    |              3.4688 |              0.032  |                0.3139 |                0.0034 |            0.177  |            0.0134 |
| E_pi85_occupation  |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| E_pi85_occupation  |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MAR_education      |      180 | rabmv           |              0.1102 |              0.0041 |                0.002  |                0.0002 |            0.0752 |            0.0032 |
| MAR_education      |      180 | rabmv_impute    |              0.1692 |              0.0076 |               -0.0006 |                0.0002 |            0.0748 |            0.0031 |
| MAR_education      |      180 | rabmv_indicator |              0.1803 |              0.007  |               -0.0009 |                0.0004 |            0.0761 |            0.0031 |
| MAR_education      |      180 | rabmv_mice      |              0.0816 |              0.0027 |               -0.0002 |                0.0001 |            0.0791 |            0.003  |
| MAR_education      |      180 | rabmv_pattern   |              0.331  |              0.0068 |                0.0105 |                0.0008 |            0.0768 |            0.0037 |
| MAR_education      |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MAR_education      |      180 | ebm_native      |              0.1706 |              0.0068 |               -0.001  |                0.0003 |            0.0874 |            0.0045 |
| MAR_education      |      180 | ebm_marginalize |              0.0909 |              0.0031 |                0.0012 |                0.0002 |            0.0874 |            0.0045 |
| MAR_education      |      180 | ebm_impute      |              0.1793 |              0.0076 |               -0.0004 |                0.0002 |            0.1065 |            0.005  |
| MAR_education      |      180 | ebm_indicator   |              0.1885 |              0.0068 |               -0.0009 |                0.0004 |            0.1064 |            0.0047 |
| MAR_education      |      180 | ebm_mice        |              0.0901 |              0.0027 |               -0.0003 |                0.0001 |            0.0986 |            0.0041 |
| MAR_education      |      180 | ebm_pattern     |              0.3464 |              0.0084 |                0.0087 |                0.0008 |            0.1435 |            0.0067 |
| MAR_education      |      180 | trees_native    |              0.817  |              0.0101 |               -0.006  |                0.001  |            0.2279 |            0.0104 |
| MAR_education      |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MAR_education      |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MAR_occupation     |      180 | rabmv           |              0.1248 |              0.0041 |                0.0025 |                0.0002 |            0.0952 |            0.0065 |
| MAR_occupation     |      180 | rabmv_impute    |              0.2054 |              0.0079 |               -0.001  |                0.0003 |            0.114  |            0.0071 |
| MAR_occupation     |      180 | rabmv_indicator |              0.184  |              0.0068 |               -0.0003 |                0.0004 |            0.0999 |            0.0067 |
| MAR_occupation     |      180 | rabmv_mice      |              0.0999 |              0.0028 |               -0      |                0.0002 |            0.1223 |            0.0066 |
| MAR_occupation     |      180 | rabmv_pattern   |              0.3325 |              0.0068 |                0.0112 |                0.0008 |            0.1317 |            0.0089 |
| MAR_occupation     |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MAR_occupation     |      180 | ebm_native      |              0.1721 |              0.0068 |               -0.0007 |                0.0003 |            0.1226 |            0.0068 |
| MAR_occupation     |      180 | ebm_marginalize |              0.1018 |              0.0032 |                0.0015 |                0.0002 |            0.1226 |            0.0068 |
| MAR_occupation     |      180 | ebm_impute      |              0.2133 |              0.0079 |               -0.001  |                0.0003 |            0.1101 |            0.007  |
| MAR_occupation     |      180 | ebm_indicator   |              0.1895 |              0.0066 |               -0.0004 |                0.0004 |            0.0911 |            0.0064 |
| MAR_occupation     |      180 | ebm_mice        |              0.1043 |              0.0027 |               -0.0002 |                0.0002 |            0.1145 |            0.0066 |
| MAR_occupation     |      180 | ebm_pattern     |              0.3514 |              0.0083 |                0.0093 |                0.0008 |            0.1193 |            0.0088 |
| MAR_occupation     |      180 | trees_native    |              0.8025 |              0.01   |               -0.0051 |                0.0011 |            0.128  |            0.0097 |
| MAR_occupation     |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MAR_occupation     |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MNAR_education     |      180 | rabmv           |              0.1082 |              0.004  |                0.0015 |                0.0002 |            0.0737 |            0.0033 |
| MNAR_education     |      180 | rabmv_impute    |              0.1651 |              0.0074 |               -0.0007 |                0.0003 |            0.0741 |            0.0031 |
| MNAR_education     |      180 | rabmv_indicator |              0.1791 |              0.0066 |               -0.0011 |                0.0004 |            0.0749 |            0.0031 |
| MNAR_education     |      180 | rabmv_mice      |              0.0795 |              0.0026 |               -0.0004 |                0.0001 |            0.0767 |            0.0029 |
| MNAR_education     |      180 | rabmv_pattern   |              0.3424 |              0.0077 |                0.0094 |                0.0008 |            0.0735 |            0.0036 |
| MNAR_education     |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MNAR_education     |      180 | ebm_native      |              0.1657 |              0.0066 |               -0.0009 |                0.0003 |            0.0879 |            0.0044 |
| MNAR_education     |      180 | ebm_marginalize |              0.0896 |              0.0031 |                0.001  |                0.0001 |            0.0879 |            0.0044 |
| MNAR_education     |      180 | ebm_impute      |              0.1757 |              0.0074 |               -0.0005 |                0.0002 |            0.1073 |            0.0051 |
| MNAR_education     |      180 | ebm_indicator   |              0.1817 |              0.0066 |               -0.0007 |                0.0003 |            0.1078 |            0.005  |
| MNAR_education     |      180 | ebm_mice        |              0.089  |              0.0026 |               -0.0004 |                0.0001 |            0.0951 |            0.004  |
| MNAR_education     |      180 | ebm_pattern     |              0.3441 |              0.0091 |                0.0091 |                0.0007 |            0.1437 |            0.0057 |
| MNAR_education     |      180 | trees_native    |              0.8259 |              0.0094 |               -0.0058 |                0.0011 |            0.2173 |            0.0104 |
| MNAR_education     |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MNAR_education     |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MNAR_occupation    |      180 | rabmv           |              0.1317 |              0.0039 |                0.0028 |                0.0003 |            0.1    |            0.007  |
| MNAR_occupation    |      180 | rabmv_impute    |              0.2455 |              0.008  |                0.0015 |                0.0003 |            0.1009 |            0.0068 |
| MNAR_occupation    |      180 | rabmv_indicator |              0.2274 |              0.0077 |                0.0014 |                0.0005 |            0.1056 |            0.0069 |
| MNAR_occupation    |      180 | rabmv_mice      |              0.1089 |              0.0027 |               -0.0004 |                0.0002 |            0.1434 |            0.0072 |
| MNAR_occupation    |      180 | rabmv_pattern   |              0.3433 |              0.0067 |                0.0101 |                0.0008 |            0.1373 |            0.009  |
| MNAR_occupation    |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MNAR_occupation    |      180 | ebm_native      |              0.2199 |              0.0076 |                0.0012 |                0.0004 |            0.1268 |            0.0071 |
| MNAR_occupation    |      180 | ebm_marginalize |              0.108  |              0.003  |                0.0019 |                0.0002 |            0.1268 |            0.0071 |
| MNAR_occupation    |      180 | ebm_impute      |              0.256  |              0.0077 |                0.0014 |                0.0003 |            0.0931 |            0.0065 |
| MNAR_occupation    |      180 | ebm_indicator   |              0.2308 |              0.0075 |                0.0015 |                0.0004 |            0.0975 |            0.0069 |
| MNAR_occupation    |      180 | ebm_mice        |              0.1133 |              0.0026 |               -0.0004 |                0.0002 |            0.1353 |            0.0074 |
| MNAR_occupation    |      180 | ebm_pattern     |              0.36   |              0.0082 |                0.0103 |                0.0009 |            0.1225 |            0.0093 |
| MNAR_occupation    |      180 | trees_native    |              0.8476 |              0.0105 |               -0.0048 |                0.0011 |            0.1268 |            0.0115 |
| MNAR_occupation    |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| MNAR_occupation    |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | rabmv           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | rabmv_impute    |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | rabmv_indicator |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | rabmv_mice      |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | rabmv_pattern   |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | ebm_native      |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | ebm_marginalize |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | ebm_impute      |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | ebm_indicator   |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | ebm_mice        |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | ebm_pattern     |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | trees_native    |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference      |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | rabmv           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | rabmv_impute    |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | rabmv_indicator |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | rabmv_mice      |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | rabmv_pattern   |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | drop            |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | ebm_native      |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | ebm_marginalize |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | ebm_impute      |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | ebm_indicator   |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | ebm_mice        |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | ebm_pattern     |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | trees_native    |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | naive           |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |
| no_difference_year |      180 | oracle          |            nan      |            nan      |              nan      |              nan      |          nan      |          nan      |

## Audits

| condition          |   events |   n |   shukla_overall_FAIL |   shukla_overall_CAUTION |   shukla_overall_PASS |   shukla_A_form_FAIL |   shukla_A_year_FAIL |   shukla_B_FAIL |   shukla_C_FAIL |   shukla_A_form_V |   shukla_A_year_V |   shukla_B_gain |   attr_auroc_iid |   attr_auroc_year_blocked |   attr_auroc_residualized |   attr_auroc_form_blocked |   attr_form_blocked_undefined |   missing_test_detects_gap |   missing_test_any_honest_alert |   checks_undeclared_detects_gap |   checks_undeclared_any_honest_alert |   checks_declared_detects_gap |   checks_declared_any_honest_alert |
|:-------------------|---------:|----:|----------------------:|-------------------------:|----------------------:|---------------------:|---------------------:|----------------:|----------------:|------------------:|------------------:|----------------:|-----------------:|--------------------------:|--------------------------:|--------------------------:|------------------------------:|---------------------------:|--------------------------------:|--------------------------------:|-------------------------------------:|------------------------------:|-----------------------------------:|
| E_pi100_education  |       60 | 100 |                  1    |                     0    |                  0    |                    1 |                 0    |            1    |               0 |             1     |             0.051 |           0.172 |            0.828 |                     0.825 |                     0     |                   nan     |                             1 |                       1    |                               0 |                               1 |                                 0.26 |                          0    |                               0.26 |
| E_pi100_education  |      180 | 100 |                  1    |                     0    |                  0    |                    1 |                 0.01 |            1    |               0 |             1     |             0.047 |           0.164 |            0.836 |                     0.836 |                     0     |                   nan     |                             1 |                       1    |                               0 |                               1 |                                 0.11 |                          0    |                               0.11 |
| E_pi100_occupation |       60 | 100 |                  1    |                     0    |                  0    |                    1 |                 0    |            1    |               0 |             1     |             0.051 |           0.168 |            0.832 |                     0.828 |                     0     |                   nan     |                             1 |                       1    |                               0 |                               1 |                                 0.15 |                          0.14 |                               0.15 |
| E_pi100_occupation |      180 | 100 |                  1    |                     0    |                  0    |                    1 |                 0.01 |            1    |               0 |             1     |             0.047 |           0.159 |            0.841 |                     0.839 |                     0     |                   nan     |                             1 |                       1    |                               0 |                               1 |                                 0.09 |                          0.04 |                               0.09 |
| E_pi50_education   |      180 | 100 |                  0    |                     0.07 |                  0.93 |                    0 |                 0.01 |            0.06 |               0 |             0.015 |             0.047 |          -0.003 |            0.838 |                     0.838 |                     0.836 |                     0.832 |                             0 |                       0.01 |                               0 |                               1 |                                 0    |                          0    |                               0    |
| E_pi50_occupation  |      180 | 100 |                  0    |                     0.07 |                  0.93 |                    0 |                 0.01 |            0.06 |               0 |             0.015 |             0.047 |          -0.003 |            0.838 |                     0.837 |                     0.835 |                     0.831 |                             0 |                       0.01 |                               0 |                               1 |                                 0    |                          0    |                               0    |
| E_pi85_education   |      180 | 100 |                  1    |                     0    |                  0    |                    1 |                 0.01 |            1    |               0 |             0.458 |             0.047 |           0.076 |            0.86  |                     0.859 |                     0.484 |                     0.355 |                             0 |                       1    |                               0 |                               1 |                                 0.15 |                          0    |                               0.03 |
| E_pi85_occupation  |      180 | 100 |                  1    |                     0    |                  0    |                    1 |                 0.01 |            1    |               0 |             0.458 |             0.047 |           0.074 |            0.861 |                     0.861 |                     0.49  |                     0.359 |                             0 |                       1    |                               0 |                               1 |                                 0.11 |                          0    |                               0.03 |
| MAR_education      |      180 | 100 |                  0    |                     0.07 |                  0.93 |                    0 |                 0.01 |            0.06 |               0 |             0.015 |             0.047 |          -0.003 |            0.838 |                     0.838 |                     0.836 |                     0.832 |                             0 |                       0.18 |                               0 |                               0 |                                 0    |                          0    |                               0    |
| MAR_occupation     |      180 | 100 |                  0    |                     0.07 |                  0.93 |                    0 |                 0.01 |            0.06 |               0 |             0.015 |             0.047 |          -0.003 |            0.837 |                     0.837 |                     0.834 |                     0.83  |                             0 |                       0.18 |                               0 |                               0 |                                 0    |                          0    |                               0    |
| MNAR_education     |      180 | 100 |                  0    |                     0.07 |                  0.93 |                    0 |                 0.01 |            0.06 |               0 |             0.015 |             0.047 |          -0.003 |            0.838 |                     0.838 |                     0.836 |                     0.832 |                             0 |                       0.01 |                               0 |                               0 |                                 0    |                          0    |                               0    |
| MNAR_occupation    |      180 | 100 |                  0    |                     0.07 |                  0.93 |                    0 |                 0.01 |            0.06 |               0 |             0.015 |             0.047 |          -0.003 |            0.838 |                     0.838 |                     0.835 |                     0.832 |                             0 |                       0.07 |                               0 |                               0 |                                 0    |                          0    |                               0    |
| no_difference      |      180 | 100 |                  0    |                     0.01 |                  0.99 |                    0 |                 0    |            0.01 |               0 |           nan     |             0.047 |          -0.002 |            0.871 |                     0.871 |                     0.869 |                   nan     |                             1 |                     nan    |                               0 |                             nan |                                 0    |                        nan    |                               0    |
| no_difference_year |      180 | 100 |                  0.14 |                     0.12 |                  0.74 |                    0 |                 0.2  |            0.2  |               0 |           nan     |             0.086 |           0.002 |            0.868 |                     0.866 |                     0.85  |                   nan     |                             1 |                     nan    |                               0 |                             nan |                                 0    |                        nan    |                               0    |

