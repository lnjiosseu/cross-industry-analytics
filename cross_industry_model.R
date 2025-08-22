# ---- Clear environment ----
rm(list = ls())

# ---- Libraries ----
library(tidyverse)
library(caret)
library(pROC)
library(reshape2)

# ---- Paths ----
data_path <- "/Users/caliboi/Desktop/Resumes/Github/Project 5/industry_modeling_data.csv"
base_dir  <- dirname(data_path)
out_dir   <- file.path(base_dir, "dashboards")
dir.create(out_dir, showWarnings = FALSE)

# ---- Load data ----
data <- read.csv(data_path)
cat("✅ Data loaded. Rows:", nrow(data), "Columns:", ncol(data), "\n")

# ---- Preprocessing ----
data$industry <- as.factor(data$industry)
data$churn <- as.factor(data$churn)

# ---- Train/Test Split ----
set.seed(42)
train_idx <- createDataPartition(data$churn, p = 0.8, list = FALSE)
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

# ---- Fit Logistic Regression ----
model <- glm(churn ~ monthly_spend + transactions + tenure_months + complaints + late_payments + industry,
             data = train_data, family = binomial)

# ---- Predictions ----
test_data$prob <- predict(model, newdata = test_data, type = "response")
test_data$pred_class <- ifelse(test_data$prob >= 0.5, 1, 0)

# ---- ROC & AUC ----
roc_obj <- roc(as.numeric(as.character(test_data$churn)), test_data$prob, quiet = TRUE)
auc_val <- as.numeric(auc(roc_obj))
gini_val <- 2 * auc_val - 1

# ---- Accuracy & Confusion Matrix ----
conf_mat <- confusionMatrix(as.factor(test_data$pred_class), test_data$churn, positive = "1")
acc_val <- conf_mat$overall["Accuracy"]

# ---- 📊 Visual 1: ROC Curve ----
png(file.path(out_dir, "cross_industry_roc_curve.png"), width = 800, height = 500, bg = "white")  # ✅ bg=white
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve - Cross-Industry Churn Model")
abline(a = 0, b = 1, lty = 2, col = "grey")
dev.off()

plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve - Cross-Industry Churn Model")
abline(a = 0, b = 1, lty = 2, col = "grey")

# ---- 📊 Visual 2: KS Statistic Curve ----
df_pred <- test_data %>%
  arrange(desc(prob)) %>%
  mutate(
    churn_num = as.numeric(as.character(churn)),
    cum_total = row_number(),
    cum_good = cumsum(ifelse(churn_num == 0, 1, 0)),
    cum_bad  = cumsum(ifelse(churn_num == 1, 1, 0))
  )

total_good <- sum(df_pred$churn_num == 0)
total_bad  <- sum(df_pred$churn_num == 1)

df_pred <- df_pred %>%
  mutate(
    pct_total = cum_total / nrow(test_data),
    TPR = cum_bad / total_bad,
    FPR = cum_good / total_good,
    KS  = abs(TPR - FPR)
  )

ks_stat <- max(df_pred$KS)

p2 <- ggplot(df_pred, aes(x = pct_total)) +
  geom_line(aes(y = TPR, color = "Churned (TPR)"), size = 1) +
  geom_line(aes(y = FPR, color = "Non-Churned (FPR)"), size = 1) +
  geom_vline(xintercept = df_pred$pct_total[which.max(df_pred$KS)], linetype = "dashed", color = "black") +
  geom_text(aes(x = df_pred$pct_total[which.max(df_pred$KS)], 
                y = 0.5, label = paste("KS =", round(ks_stat, 3))),
            angle = 90, vjust = -0.5, hjust = -0.1, size = 5) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "KS Statistic Curve - Cross-Industry Model",
       x = "Proportion of Population", y = "Cumulative Percentage",
       color = "Legend") +
  theme_minimal() +
  theme(                                                     # ✅ white plot/panel
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA)
  )

print(p2)
ggsave(filename = file.path(out_dir, "cross_industry_ks_curve.png"), plot = p2,
       width = 9, height = 5, dpi = 300, bg = "white")                              # ✅ bg=white

# ---- 📊 Visual 3: Feature Importance ----
coef_df <- summary(model)$coefficients %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  mutate(abs_coef = abs(Estimate)) %>%
  arrange(desc(abs_coef))

p3 <- ggplot(coef_df %>% head(10), aes(x = reorder(feature, abs_coef), y = abs_coef)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Top 10 by Coefficient Magnitude)",
       x = "Feature", y = "Absolute Coefficient") +
  theme_minimal() +
  theme(                                                     # ✅ white plot/panel
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA)
  )

print(p3)
ggsave(filename = file.path(out_dir, "cross_industry_feature_importance.png"), plot = p3,
       width = 9, height = 5, dpi = 300, bg = "white")                              # ✅ bg=white

# ---- 📊 Visual 4: Churn Probability Distribution by Industry ----
p4 <- ggplot(test_data, aes(x = prob, fill = churn)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_wrap(~industry, scales = "free_y") +
  labs(title = "Predicted Churn Probability Distribution by Industry",
       x = "Predicted Probability of Churn", y = "Count", fill = "Churn") +
  theme_minimal() +
  theme(                                                     # ✅ white plot/panel
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA)
  )

print(p4)
ggsave(filename = file.path(out_dir, "cross_industry_churn_distribution.png"), plot = p4,
       width = 12, height = 8, dpi = 300, bg = "white")                             # ✅ bg=white

# ---- 📊 Industry-Level Performance ----
industry_perf <- test_data %>%
  group_by(industry) %>%
  group_modify(~ {
    churn_vec <- as.numeric(as.character(.x$churn))   # ✅ ensure numeric 0/1
    prob_vec  <- .x$prob
    pred_vec  <- .x$pred_class
    
    auc_val <- tryCatch({
      as.numeric(auc(roc(churn_vec, prob_vec, quiet = TRUE)))  # ✅ force numeric
    }, error = function(e) NA)
    
    tibble(
      n = nrow(.x),
      churn_rate = mean(churn_vec),
      accuracy = mean(pred_vec == churn_vec),
      auc = auc_val
    )
  }) %>%
  ungroup()

print(industry_perf)

# ---- 📊 Visual 5: Heatmap of Industry Performance ----
industry_melt <- reshape2::melt(industry_perf %>% select(industry, churn_rate, accuracy, auc), id.vars = "industry")

industry_order <- industry_perf %>%
  arrange(desc(auc)) %>%
  pull(industry)

industry_melt$industry <- factor(industry_melt$industry, levels = industry_order)

p5 <- ggplot(industry_melt, aes(x = variable, y = industry, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), color = "black") +
  scale_fill_gradient(low = "red", high = "green") +
  labs(title = "Industry-Level Performance Heatmap (Sorted by AUC)",
       x = "Metric", y = "Industry", fill = "Value") +
  theme_minimal() +
  theme(                                                     # ✅ white plot/panel
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA)
  )

print(p5)
ggsave(filename = file.path(out_dir, "cross_industry_heatmap.png"), plot = p5,
       width = 9, height = 6, dpi = 300, bg = "white")                              # ✅ bg=white

# ---- 📄 Summary Report ----
report_path <- file.path(out_dir, "cross_industry_model_summary.txt")

sink(report_path)
cat("🔎 Cross-Industry Churn Model Summary\n")
cat("------------------------------------\n")
cat("Train records:", nrow(train_data), "\n")
cat("Test records:", nrow(test_data), "\n\n")

cat("Logistic Regression Model Coefficients:\n")
print(summary(model)$coefficients)
cat("\n")

cat("Performance Metrics (Overall):\n")
cat("AUC:", round(auc_val, 3), "\n")
cat("GINI:", round(gini_val, 3), "\n")
cat("Accuracy:", round(acc_val, 3), "\n")
cat("KS Statistic:", round(ks_stat, 3), "\n\n")

cat("Confusion Matrix:\n")
print(conf_mat$table)
cat("\n")

cat("📊 Industry-Level Performance (sorted by AUC):\n")
print(industry_perf %>% arrange(desc(auc)))
sink()

cat("✅ All visuals + 📄 summary saved in:", out_dir, "\n")
