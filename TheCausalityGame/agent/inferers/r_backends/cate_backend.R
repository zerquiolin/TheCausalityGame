args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: cate_backend.R INPUT_JSON OUTPUT_JSON")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("R package jsonlite is required")

payload <- jsonlite::fromJSON(args[[1]], simplifyVector = TRUE)
expected_method <- Sys.getenv("TCG_R_BACKEND_METHOD")
if (nzchar(expected_method) && payload$method != expected_method) {
  stop(paste("R backend entrypoint expected", expected_method, "but received", payload$method))
}
train <- as.data.frame(payload$train, check.names = FALSE)
query <- as.data.frame(payload$query, check.names = FALSE)
features <- as.character(payload$features)
outcome <- as.character(payload$outcome)
w <- as.numeric(payload$treatment)
params <- payload$params
set.seed(as.integer(params$seed))

package_meta <- function(package, commit) {
  list(
    package = package,
    package_version = as.character(utils::packageVersion(package)),
    pinned_commit = commit,
    implementation_kind = "authoritative"
  )
}

if (payload$method == "s_learner") {
  if (!requireNamespace("causalToolbox", quietly = TRUE)) stop("R package causalToolbox is required")
  model <- causalToolbox::S_RF(
    feat = train[, features, drop = FALSE],
    tr = w,
    yobs = as.numeric(train[[outcome]]),
    nthread = as.integer(params$nthread)
  )
  predictions <- causalToolbox::EstimateCate(model, query[, features, drop = FALSE])
  backend <- package_meta("causalToolbox", "75acc74ae689fddebe63a43597ff21f3ec233bd4")
} else {
  if (!requireNamespace("causalTree", quietly = TRUE)) stop("R package causalTree is required")
  formula <- stats::reformulate(features, response = outcome)
  control <- rpart::rpart.control(cp = as.numeric(params$cp), xval = as.integer(params$xval))
  propensity <- mean(w)

  if (payload$method == "transformed_outcome") {
    model <- causalTree::causalTree(
      formula,
      data = train,
      treatment = w,
      split.Rule = "TOT",
      split.Honest = FALSE,
      split.Bucket = isTRUE(params$split_bucket),
      cv.option = "TOT",
      cv.Honest = FALSE,
      minsize = as.integer(params$minsize),
      propensity = propensity,
      control = control
    )
  } else if (payload$method == "honest_causal_tree") {
    treated <- which(w == 1)
    controls <- which(w == 0)
    fraction <- as.numeric(params$honest_fraction)
    train_idx <- c(
      sample(treated, max(1, floor(length(treated) * (1 - fraction)))),
      sample(controls, max(1, floor(length(controls) * (1 - fraction))))
    )
    estimate_idx <- setdiff(seq_len(nrow(train)), train_idx)
    if (length(unique(w[train_idx])) != 2 || length(unique(w[estimate_idx])) != 2) {
      stop("honest causal tree requires both treatment arms in split and estimation samples")
    }
    model <- causalTree::honest.causalTree(
      formula,
      data = train[train_idx, , drop = FALSE],
      treatment = w[train_idx],
      est_data = train[estimate_idx, , drop = FALSE],
      est_treatment = w[estimate_idx],
      split.Rule = "CT",
      split.Honest = TRUE,
      HonestSampleSize = length(estimate_idx),
      split.Bucket = isTRUE(params$split_bucket),
      cv.option = "CT",
      cv.Honest = TRUE,
      minsize = as.integer(params$minsize),
      propensity = propensity,
      control = control
    )
  } else {
    stop(paste("unsupported CATE backend method", payload$method))
  }

  if (!is.null(model$cptable) && nrow(model$cptable) > 0) {
    cp_index <- which.min(model$cptable[, "xerror"])
    model <- rpart::prune(model, cp = model$cptable[cp_index, "CP"])
  }
  predictions <- stats::predict(model, newdata = query[, features, drop = FALSE])
  backend <- package_meta("causalTree", "51a8762bff0ab17b971bce852cb37bece2d8240f")
}

jsonlite::write_json(
  list(predictions = as.numeric(predictions), backend = backend),
  args[[2]],
  auto_unbox = TRUE,
  digits = NA
)
