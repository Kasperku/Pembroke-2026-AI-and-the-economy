# ===========================================================================
# Event Study: AI Exposure and Metro Migration
#
# Y_mt = alpha_m + alpha_t + sum_k beta_k (AIExposure_m x Year_k) + X_mt'gamma + e_mt
# Base year: 2019 | SEs clustered at metro
# Pre-treatment: 2014-2022 | Post-treatment: 2023-2024
# ===========================================================================

library(fixest)
library(data.table)
library(ggplot2)

# ===========================================================================
# CONFIGURATION — choose which sample(s) to run
# ===========================================================================
# Set to TRUE to run that sample specification.
# You can run one, several, or all three.

RUN_ALL_YEARS       <- TRUE   # includes 2020 and 2021
RUN_DROP_2020       <- TRUE   # drops 2020 (experimental Census weights)
RUN_DROP_2020_2021  <- TRUE   # drops 2020 + 2021 (COVID recovery) — main spec

# ===========================================================================
# Sample definitions
# ===========================================================================
SAMPLE_SPECS <- list()

if (RUN_ALL_YEARS) {
  SAMPLE_SPECS[["all_years"]] <- list(
    label      = "All years (incl. 2020 & 2021)",
    drop_years = c()
  )
}
if (RUN_DROP_2020) {
  SAMPLE_SPECS[["drop_2020"]] <- list(
    label      = "Drop 2020 (experimental weights)",
    drop_years = c(2020)
  )
}
if (RUN_DROP_2020_2021) {
  SAMPLE_SPECS[["drop_2020_2021"]] <- list(
    label      = "Drop 2020 & 2021 (main spec)",
    drop_years = c(2020, 2021)
  )
}

# ===========================================================================
# Outcomes and controls (shared across all specs)
# ===========================================================================
outcomes <- list(
  "In-migration rate"          = "inmig_rate",
  "In-migration rate (BA+)"    = "inmig_rate_ba",
  "In-migration rate (no BA)"  = "inmig_rate_no_ba",
  "Skilled in-migrant share"   = "skilled_inmig_share",
  "Net migration BA+ (rate)"   = "net_mig_ba_rate",
  "Net migration no BA (rate)" = "net_mig_no_ba_rate",
  "Incumbent share BA"         = "incumbent_share_ba",
  "In-mig rate young (25-35)"  = "inmig_rate_young",
  "In-mig rate old (45-55)"    = "inmig_rate_old"
)

controls     <- c("share_ba", "log_pop", "mean_age", "share_female_lf", "unemp_rate")
ctrl_formula <- paste(controls, collapse = " + ")

# ===========================================================================
# Load base panel (once)
# ===========================================================================
SCRIPT_DIR  <- dirname(sys.frame(1)$ofile)
PANEL_PATH  <- file.path(SCRIPT_DIR, "..", "Data", "Processed", "metro_year_panel.csv")
base_panel  <- fread(PANEL_PATH)
base_panel  <- base_panel[!is.na(ai_exposure)]

# ===========================================================================
# Helper: run all regressions for a given panel subset
# ===========================================================================
run_spec <- function(panel, spec_name, spec_label, outdir) {

  cat("\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")
  cat(sprintf("SPEC: %s\n", spec_label))
  cat(sprintf("Years in sample: %s\n", paste(sort(unique(panel$year)), collapse = ", ")))
  cat(sprintf("Obs: %d metros x %d years = %d rows\n",
              uniqueN(panel$met2013), uniqueN(panel$year), nrow(panel)))
  cat(sprintf("AI exposure range: [%.3f, %.3f], sd = %.3f\n",
              min(panel$ai_exposure), max(panel$ai_exposure), sd(panel$ai_exposure)))
  cat(paste(rep("=", 70), collapse = ""), "\n")

  panel <- copy(panel)
  panel[, year_f := relevel(factor(year), ref = "2019")]
  panel[, ai_std := (ai_exposure - mean(ai_exposure)) / sd(ai_exposure)]
  panel[, post   := as.integer(year >= 2023)]

  spec_outdir <- file.path(outdir, spec_name)
  dir.create(spec_outdir, recursive = TRUE, showWarnings = FALSE)

  results <- list()

  # --- Event study regressions ---
  for (name in names(outcomes)) {
    yvar <- outcomes[[name]]
    if (all(is.na(panel[[yvar]]))) {
      cat(sprintf("  Skipping %s (all NA)\n", name))
      next
    }

    fml <- as.formula(paste0(
      yvar, " ~ i(year_f, ai_std, ref = '2019') + ", ctrl_formula,
      " | met2013 + year"
    ))
    est <- feols(fml, data = panel, cluster = ~met2013)
    results[[name]] <- est

    cat(sprintf("\n=== %s ===\n", name))
    print(summary(est))

    # Event study plot
    pdf(file.path(spec_outdir, paste0("es_", yvar, ".pdf")), width = 8, height = 5)
    iplot(est,
          main = paste0(name, "\n(", spec_label, ")"),
          xlab = "Year",
          ylab = "Coefficient (1 SD AI exposure)")
    abline(v = 2022.5, lty = 2, col = "red")
    dev.off()
  }

  # --- Summary table: post-treatment coefficients ---
  cat("\n--- Post-treatment coefficients (2023, 2024) ---\n")
  cat(sprintf("%-35s %10s %10s\n", "Outcome", "2023", "2024"))
  cat(paste(rep("-", 60), collapse = ""), "\n")

  get_coef <- function(est, yr) {
    ct       <- coeftable(est)
    row_name <- paste0("year_f::", yr, ":ai_std")
    if (row_name %in% rownames(ct)) {
      val   <- ct[row_name, "Estimate"]
      pv    <- ct[row_name, "Pr(>|t|)"]
      stars <- ifelse(pv < 0.01, "***", ifelse(pv < 0.05, "**", ifelse(pv < 0.1, "*", "")))
      sprintf("%8.4f%s", val, stars)
    } else {
      sprintf("%8s", "--")
    }
  }

  for (name in names(results)) {
    cat(sprintf("%-35s %10s %10s\n",
                name, get_coef(results[[name]], 2023), get_coef(results[[name]], 2024)))
  }
  cat("Significance: *** p<0.01, ** p<0.05, * p<0.1\n")

  # --- Simple DID robustness ---
  cat("\n--- Simple DID: Pre (up to 2022) vs Post (2023-2024) ---\n")
  for (name in names(outcomes)) {
    yvar <- outcomes[[name]]
    if (all(is.na(panel[[yvar]]))) next

    fml <- as.formula(paste0(
      yvar, " ~ post:ai_std + ", ctrl_formula, " | met2013 + year"
    ))
    est <- feols(fml, data = panel, cluster = ~met2013)
    ct  <- coeftable(est)

    if ("post:ai_std" %in% rownames(ct)) {
      val   <- ct["post:ai_std", "Estimate"]
      se    <- ct["post:ai_std", "Std. Error"]
      pv    <- ct["post:ai_std", "Pr(>|t|)"]
      stars <- ifelse(pv < 0.01, "***", ifelse(pv < 0.05, "**", ifelse(pv < 0.1, "*", "")))
      cat(sprintf("  %-35s  β = %8.5f (%7.5f) %s\n", name, val, se, stars))
    }
  }
}

# ===========================================================================
# Run all selected specs
# ===========================================================================
outdir  <- file.path(SCRIPT_DIR, "output")
logfile <- file.path(outdir, "results.txt")
dir.create(outdir, showWarnings = FALSE)

sink(logfile, split = TRUE)

cat("===========================================================================\n")
cat("Event Study: AI Exposure and Metro Migration\n")
cat(sprintf("Run date: %s\n", Sys.time()))
cat("===========================================================================\n")

for (spec_name in names(SAMPLE_SPECS)) {
  spec  <- SAMPLE_SPECS[[spec_name]]
  panel <- base_panel[!year %in% spec$drop_years]
  run_spec(panel, spec_name, spec$label, outdir)
}

sink()
cat(sprintf("\nAll results saved to %s\n", logfile))
