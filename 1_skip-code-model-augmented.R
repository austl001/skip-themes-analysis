
# Setup -------------------------------------------------------------------

library(tidyverse)
library(tidymodels)


# Import and check data ---------------------------------------------------

skip_themes_augmented <- readRDS(paste0(getwd(), "/data/skip_themes_augmented.RDS"))
glimpse(skip_themes_augmented)
skimr::skim(skip_themes_augmented)

# Count missing and uncategorised skips to filter outliers

skip_themes_augmented %>% 
  count(Wholesaler, SkipCodeTheme) %>% 
  group_by(Wholesaler) %>% 
  mutate(
    SkipCount = sum(n),
    ThemeProp = n / SkipCount
  ) %>% 
  filter(SkipCodeTheme %in% c("Uncategorised reason", "Missing reason")) %>% 
  ggplot(aes(reorder(Wholesaler, n), n)) +
  geom_col(aes(fill = SkipCodeTheme)) + 
  coord_flip()


# Refining the model dataset

skips_model_df_all <- skip_themes_augmented %>%
  mutate(
    InitialMeterReadDate = 
      as.Date(replace_na(
        as.character(InitialMeterReadDate), "2017-04-01")
        ),
    LatestReadInSystem = if_else(
      is.na(LatestReadInSystem), InitialMeterReadDate, LatestReadInSystem),
    DaysSinceRead = as.numeric(Sys.Date() - LatestReadInSystem),
    MeterAge = case_when(
      InitialMeterReadDate <= "2017-04-01" ~ "Pre-market",
      InitialMeterReadDate > "2017-04-01" ~ "Post-market"
    ),
    FailureSinceRead = InitiatingEventDate - LatestReadInSystem,
    LumStatus = factor(
      case_when(
        DaysSinceRead / 365 < 1 ~ "Read", 
        TRUE ~ "Long unread"
      )
    ),
    MeterSizeBanding = case_when(
      PhysicalMeterSize <= 15 ~ "Small",
      PhysicalMeterSize > 15 & PhysicalMeterSize <= 50 ~ "Medium",
      PhysicalMeterSize > 50 ~ "Large",
      TRUE ~ "Missing"
      ),
    GisIssues = case_when(
      DuplicatedGis == 1 ~ "Yes",
      Over20MetersStacked == 1 ~ "Yes",
      PostcodeCentre == 1 ~ "Yes",
      UprnCentre == 1 ~ "Yes",
      FarFromPostcodeCentre == 1 ~ "Yes",
      FarFromUprnCentre == 1 ~ "Yes",
      TRUE ~ "No"
      ),
    SpidCount = case_when(
      SpiDs <= 1000 ~ "Small",
      SpiDs > 1000 & SpiDs < 10000 ~ "Medium",
      SpiDs > 10000 ~ "Large"
      ),
    UprnProblems = case_when(
      UprnProblems == "No problems" ~ "No", 
      is.na(UprnProblems) ~ "Yes",
      TRUE ~ "Yes"
      ),
    UprnCompleteness = if_else(is.na(UprnCompleteness), "Incomplete", "Complete"),
    VoaCompleteness = if_else(is.na(VoaCompleteness), "Incomplete", "Complete"),
    SupplyType = if_else(str_detect(Spid, "S"), "Sewerage", "Water"),
    WholesalerId = fct_lump_min(WholesalerId, min = 100),
    SkipLength = replace_na(SkipLength, 0),
    ValidMeterManufacturers = fct_lump_min(ValidMeterManufacturers, min = 100),
    ValidMeterManufacturers = replace_na(ValidMeterManufacturers, "Other")
    ) %>%
  select(
    SkipControlAdj, SkipLength, Performance, MeterLocationCode,
    MeterReadMinimumFrequency, LumStatus, RetailerType, SpidCount, MeterAge,
    UprnCompleteness, VoaCompleteness, UprnProblems, GisIssues, MeterSizeBanding,
    StdBanding, ValidMeterManufacturers, SkipCodeTheme, Mps, RetailerIdAnon,
    WholesalerId
    ) %>%
  drop_na(MeterLocationCode, MeterReadMinimumFrequency, StdBanding) %>% 
  mutate_if(is.character, as.factor) %>%
  mutate_if(lubridate::is.difftime, as.numeric)

skips_model_df <- skips_model_df_all %>% 
  filter(!SkipCodeTheme %in% c("Missing reason", "Uncategorised reason")) %>% 
  select(-(SkipCodeTheme:WholesalerId))

glimpse(skips_model_df)
skimr::skim(skips_model_df)


# Create splits, train and test data

set.seed(234)

splits <- initial_split(skips_model_df, strata = SkipControlAdj)
skips_train <- training(splits)
skips_test <- testing(splits)

# Create recipe and check prepped data

skips_rec <- 
  recipe(SkipControlAdj ~ ., data = skips_train) %>%
  step_log(MeterAge, SkipLength, Performance) %>% 
  step_naomit(everything(), skip = TRUE) %>% 
  step_novel(all_nominal(), -all_outcomes()) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_numeric(), -all_outcomes()) %>% 
  step_corr(all_predictors(), threshold = 0.7, method = "spearman")

prepped_data <- skips_rec %>% 
  prep() %>% 
  juice()

glimpse(prepped_data)
skimr::skim(prepped_data)

# Create validation set

set.seed(345)
cv_folds <- vfold_cv(skips_train, v = 5, strata = SkipControlAdj)

# Set model and engine

log_model <- 
  logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

library(ranger)

rf_model <- 
  rand_forest() %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

library(xgboost)

xgb_model <- 
  boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

knn_model <- 
  nearest_neighbor(neighbors = 4) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")


# Create workflows

log_workflow <- 
  workflow() %>% 
  add_recipe(skips_rec) %>% 
  add_model(log_model)

rf_workflow <- 
  workflow() %>% 
  add_recipe(skips_rec) %>% 
  add_model(rf_model)

xgb_workflow <- 
  workflow() %>% 
  add_recipe(skips_rec) %>% 
  add_model(xgb_model)

knn_workflow <- 
  workflow() %>% 
  add_recipe(skips_rec) %>% 
  add_model(knn_model)


# Evaluate models

## Logistic regression

log_res <- log_workflow %>% 
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(
      recall, precision, f_meas, accuracy,
      kap, roc_auc, sens, spec
    ),
    control = control_resamples(save_pred = TRUE)
  )

log_res %>% collect_metrics(summarize = TRUE)

log_pred <- 
  log_res %>% 
  collect_predictions()
log_pred %>% 
  conf_mat(SkipControlAdj, .pred_class) %>%
  autoplot(type = "heatmap")
log_pred %>% 
  group_by(id) %>% 
  roc_curve(SkipControlAdj, `.pred_Outside retailer control`) %>%
  autoplot()


## Random forest

rf_res <- rf_workflow %>% 
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(
      recall, precision, f_meas, accuracy,
      kap, roc_auc, sens, spec
    ),
    control = control_resamples(
      save_pred = TRUE
    )
  )

rf_res %>% collect_metrics(summarise = TRUE)
rf_pred <- 
  rf_res %>% 
  collect_predictions()
rf_pred %>% 
  conf_mat(SkipControlAdj, .pred_class) %>%
  autoplot(type = "heatmap")
rf_pred %>% 
  group_by(id) %>% 
  roc_curve(SkipControlAdj, `.pred_Outside retailer control`) %>%
  autoplot()


## XGBoost

xgb_res <- xgb_workflow %>% 
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(
      recall, precision, f_meas, accuracy,
      kap, roc_auc, sens, spec
    ),
    control = control_resamples(
      save_pred = TRUE
    )
  )

xgb_res %>% collect_metrics(summarise = TRUE)
xgb_pred <- 
  xgb_res %>% 
  collect_predictions()
xgb_pred %>% 
  conf_mat(SkipControlAdj, .pred_class) %>%
  autoplot(type = "heatmap")
xgb_pred %>% 
  group_by(id) %>% 
  roc_curve(SkipControlAdj, `.pred_Outside retailer control`) %>%
  autoplot()


## KNN

knn_res <- knn_workflow %>% 
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(
      recall, precision, f_meas, accuracy,
      kap, roc_auc, sens, spec
    ),
    control = control_resamples(
      save_pred = TRUE
    )
  )

knn_res %>% collect_metrics(summarise = TRUE)
knn_pred <- 
  knn_res %>% 
  collect_predictions()
knn_pred %>% 
  conf_mat(SkipControlAdj, .pred_class) %>%
  autoplot(type = "heatmap")
knn_pred %>% 
  group_by(id) %>% 
  roc_curve(SkipControlAdj, `.pred_Outside retailer control`) %>%
  autoplot()

# Compare models

log_metrics <- log_res %>% 
  collect_metrics(summarise = TRUE) %>% 
  mutate(model = "Logistic Regression")

rf_metrics <- rf_res %>% 
  collect_metrics(summarise = TRUE) %>% 
  mutate(model = "Random Forest")

xgb_metrics <- xgb_res %>% 
  collect_metrics(summarise = TRUE) %>% 
  mutate(model = "XGBoost")

knn_metrics <- knn_res %>% 
  collect_metrics(summarise = TRUE) %>% 
  mutate(model = "Knn")

model_compare <- bind_rows(
  log_metrics, rf_metrics, xgb_metrics, knn_metrics
)

model_compare_pivot <- model_compare %>% 
  select(model, .metric, mean, std_err) %>% 
  pivot_wider(names_from = .metric, values_from = c(mean, std_err)) %>% 
  mutate(model = factor(model))

model_compare_pivot %>% 
  arrange(mean_f_meas) %>% 
  mutate(mean_f_meas) %>% 
  ggplot(aes(reorder(model, mean_f_meas), mean_f_meas, fill = model)) + 
  geom_col() + 
  coord_flip() + 
  MOSLR::scale_fill_MOSL() + 
  geom_text(aes(label = round(mean_f_meas, 3), y = mean_f_meas + 0.08), vjust = 1)

model_compare_pivot %>% 
  arrange(mean_roc_auc) %>% 
  ggplot(aes(reorder(model, mean_roc_auc), mean_roc_auc, fill = model)) + 
  geom_col() + 
  coord_flip() + 
  MOSLR::scale_fill_MOSL() + 
  geom_text(aes(label = round(mean_roc_auc, 3), y = mean_roc_auc + 0.08), vjust = 1)

model_compare_pivot %>% 
  arrange(mean_kap) %>% 
  ggplot(aes(reorder(model, mean_kap), mean_kap, fill = model)) + 
  geom_col() + 
  coord_flip() + 
  MOSLR::scale_fill_MOSL() + 
  geom_text(aes(label = round(mean_kap, 3), y = mean_kap + 0.08), vjust = 1)

model_compare_pivot %>% slice_max(mean_f_meas)


# Last fit evaluation on test set

last_fit_xgb <- last_fit(
  xgb_workflow,
  split = splits,
  metrics = metric_set(
    recall, precision, f_meas, accuracy, kap,
    roc_auc, sens, spec
  )
)

last_fit_xgb %>% collect_metrics()

library(vip)

last_fit_xgb %>% 
  pluck(".workflow", 1) %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 10)

last_fit_xgb %>% 
  collect_predictions() %>% 
  conf_mat(SkipControlAdj, .pred_class) %>% 
  autoplot(type = "heatmap")

last_fit_xgb %>% 
  collect_predictions() %>% 
  roc_curve(SkipControlAdj, '.pred_Outside retailer control') %>% 
  autoplot()


# Use model to predict retailer control

xgb_wflow_final_fit <- fit(xgb_workflow, data = skips_model_df)

skips_missing <- skips_model_df_all %>%
  filter(
    SkipCodeTheme %in% c("Missing reason", "Uncategorised reason"),
    Mps != "MPS 17"
  )

skips_missing$.pred_class <- unlist(predict(xgb_wflow_final_fit, skips_missing))


## Append results

skips_themes_modelled <- skips_model_df_all %>%
  filter(
    !SkipCodeTheme %in% c("Missing reason", "Uncategorised reason"),
    Mps != "MPS 17"
  ) %>% 
  mutate(.pred_class = "None") %>% 
  bind_rows(skips_missing) %>% 
  mutate(SkipControlAdjNew = if_else(
    .pred_class == "None", 
    as.character(SkipControlAdj), 
    .pred_class
    )
  )

table(skips_themes_modelled$SkipControlAdjNew)

skips_themes_modelled_summary <- skips_themes_modelled %>% 
  mutate(
    OutsideRetailerControl = 
      if_else(SkipControlAdjNew == "Outside retailer control", 1, 0),
    Scenario = "Predicted control"
  ) %>%
  group_by(Scenario) %>% 
  summarise(OutsideRetailerControl = mean(OutsideRetailerControl, na.rm = TRUE))

skips_themes_modelled_summary

table(skips_themes_modelled$WholesalerId, skips_themes_modelled$SkipControlAdjNew)

skips_themes_modelled_tps <- skips_themes_modelled %>% 
  mutate(
    OutsideRetailerControl = if_else(SkipControlAdj == "Outside retailer control", 1, 0)
    ) %>%
  group_by(RetailerIdAnon, RetailerType) %>%
  summarise(
    OutsideRetailerControl = mean(OutsideRetailerControl, na.rm = TRUE),
    TotalSkips = n(),
    Performance = mean(Performance, na.rm = TRUE)
  )

saveRDS(skips_themes_modelled_tps, paste0(getwd(), "/data/model_summary_tps.RDS"))      
saveRDS(skips_themes_modelled, paste0(getwd(), "/data/model_all.RDS"))        
saveRDS(skips_themes_modelled_summary, paste0(getwd(), "/data/model_summary.RDS"))        


skips_themes_modelled %>% 
  mutate(
    OutsideRetailerControl = 
      if_else(SkipControlAdjNew == "Outside retailer control", 1, 0),
    Scenario = "Predicted control"
  ) %>%
  group_by(RetailerType, RetailerIdAnon, Performance, Scenario) %>% 
  summarise(
    OutsideRetailerControl = mean(OutsideRetailerControl, na.rm = TRUE), 
    n = n()) %>% 
  ggplot(aes(Performance, OutsideRetailerControl)) + 
  geom_point(aes(colour = RetailerType)) +
  geom_smooth(method = "lm", col = "red", lty = "dashed", alpha = 0.2)# +
  facet_wrap(~RetailerType)



