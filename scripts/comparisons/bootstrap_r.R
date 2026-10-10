args <- commandArgs(trailingOnly = TRUE)
library_dir <- if (length(args) >= 1) args[[1]] else ".r-library"
dir.create(library_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(library_dir), .libPaths()))

install_cran <- function(package, version) {
  installed <- requireNamespace(package, quietly = TRUE)
  correct <- installed && as.character(utils::packageVersion(package)) == version
  if (!correct) {
    if (!requireNamespace("remotes", quietly = TRUE) && package != "remotes") {
      install.packages("remotes", repos = "https://cloud.r-project.org")
    }
    if (package == "remotes") {
      if (!installed) install.packages(package, repos = "https://cloud.r-project.org")
      if (as.character(utils::packageVersion(package)) != version) {
        remotes::install_version(package, version = version, repos = "https://cloud.r-project.org")
      }
    } else {
      remotes::install_version(package, version = version, repos = "https://cloud.r-project.org")
    }
  }
}

install_cran("remotes", "2.5.0")
install_cran("jsonlite", "1.9.1")

# Local installs do not preserve remotes' Git metadata. Verify the checkout and
# stamp the installed DESCRIPTION so subsequent runs can check the actual source.
stamp_source <- function(source_dir, commit, patch) {
  actual <- system2("git", c("-C", source_dir, "rev-parse", "HEAD"), stdout = TRUE)
  if (length(actual) != 1 || !identical(actual, commit)) {
    stop("Locked source checkout mismatch: expected ", commit, ", got ", paste(actual, collapse = " "))
  }
  path <- file.path(source_dir, "DESCRIPTION")
  description <- read.dcf(path)
  description <- cbind(description, TCGSourceCommit = commit, TCGCompatibilityPatch = patch)
  write.dcf(description, path)
}

installed_source_matches <- function(package, commit, patch) {
  if (!requireNamespace(package, quietly = TRUE)) return(FALSE)
  description <- utils::packageDescription(package)
  identical(description$TCGSourceCommit, commit) &&
    identical(description$TCGCompatibilityPatch, patch)
}

checkout_source <- function(repository, commit, prefix) {
  source_dir <- tempfile(prefix)
  commands <- list(
    c("init", "--quiet", source_dir),
    c("-C", source_dir, "fetch", "--quiet", "--depth=1",
      paste0("https://github.com/", repository, ".git"), commit),
    c("-C", source_dir, "checkout", "--quiet", "--detach", "FETCH_HEAD")
  )
  for (command in commands) {
    output <- system2("git", command, stdout = TRUE, stderr = TRUE)
    status <- attr(output, "status")
    if (!is.null(status) && status != 0) {
      stop("Could not fetch locked ", repository, " source: ", paste(output, collapse = "\n"))
    }
  }
  return(source_dir)
}

# causalTree 51a8762 predates R 4.5's removal of findVarInFrame and assumes
# pre-C23 handling of unspecified function prototypes. Patch only these build
# compatibility points; retain the locked upstream algorithm and parameters.
install_causal_tree <- function(commit) {
  patch <- "r45-public-api-c17-v1"
  if (installed_source_matches("causalTree", commit, patch)) {
    message("causalTree locked source and compatibility patch are already installed")
    return(invisible(NULL))
  }
  source_dir <- checkout_source("susanathey/causalTree", commit, "tcg-causalTree-")

  replace_once <- function(text, old, new, file) {
    matches <- gregexpr(old, text, fixed = TRUE)[[1]]
    if (matches[1] < 0 || length(matches) != 1) {
      stop("Expected one source match for causalTree compatibility patch in ", file)
    }
    sub(old, new, text, fixed = TRUE)
  }

  callback_path <- file.path(source_dir, "src", "causalTree_callback.c")
  callback <- paste(readLines(callback_path, warn = FALSE), collapse = "\n")
  callback <- replace_once(
    callback,
    "#include <Rinternals.h>",
    paste(
      "#include <Rinternals.h>",
      "#include <Rversion.h>",
      "/* R 4.5+ public replacement for the removed non-API lookup. */",
      "#if R_VERSION >= R_Version(4, 5, 0)",
      "#define ct_findVarInFrame(rho, sym) R_getVar((sym), (rho), FALSE)",
      "#else",
      "#define ct_findVarInFrame(rho, sym) findVarInFrame((rho), (sym))",
      "#endif",
      sep = "\n"
    ),
    callback_path
  )
  for (name in c("yback", "wback", "xback", "nback")) {
    callback <- replace_once(
      callback,
      paste0('findVarInFrame(rho, install("', name, '"))'),
      paste0('ct_findVarInFrame(rho, install("', name, '"))'),
      callback_path
    )
  }
  writeLines(callback, callback_path)

  header_path <- file.path(source_dir, "src", "causalTree.h")
  header <- paste(readLines(header_path, warn = FALSE), collapse = "\n")
  header <- replace_once(
    header,
    "#include <math.h>",
    paste(
      "#include <math.h>",
      "#include <stdlib.h>",
      "/* R 4.5+ no longer exposes the legacy Free macro to packages. */",
      "#ifndef Free",
      "#define Free free",
      "#endif",
      sep = "\n"
    ),
    header_path
  )
  writeLines(header, header_path)

  stamp_source(source_dir, commit, patch)

  cc <- system2(file.path(R.home("bin"), "R"), c("CMD", "config", "CC"), stdout = TRUE)
  if (length(cc) != 1 || is.na(cc)) stop("Could not determine the R C compiler")
  cc <- sub("-std=[^[:space:]]+", "-std=gnu17", cc)
  if (!grepl("-std=gnu17", cc, fixed = TRUE)) cc <- paste(cc, "-std=gnu17")
  makevars <- tempfile("tcg-causalTree-Makevars-")
  writeLines(paste("CC =", cc), makevars)
  old_makevars <- Sys.getenv("R_MAKEVARS_USER", unset = NA_character_)
  Sys.setenv(R_MAKEVARS_USER = makevars)
  on.exit({
    if (is.na(old_makevars)) Sys.unsetenv("R_MAKEVARS_USER") else Sys.setenv(R_MAKEVARS_USER = old_makevars)
    unlink(c(makevars, source_dir), recursive = TRUE)
  }, add = TRUE)
  remotes::install_local(source_dir, upgrade = "never", force = TRUE)
}

install_causal_tree("51a8762bff0ab17b971bce852cb37bece2d8240f")

install_causal_toolbox <- function(commit) {
  patch <- "srf-rforestry-api-dimensions-v1"
  if (installed_source_matches("causalToolbox", commit, patch)) {
    srf <- paste(
      deparse(body(get("S_RF_fully_specified", asNamespace("causalToolbox")))),
      collapse = " "
    )
    if (grepl("drop = FALSE", srf, fixed = TRUE) &&
        !grepl("OOBhonest = hyperparameter_list", srf, fixed = TRUE)) {
      message("causalToolbox S_RF compatibility patch is already installed")
      return(invisible(NULL))
    }
  }
  if (!requireNamespace("causalToolbox", quietly = TRUE)) {
    remotes::install_github(
      paste0("forestry-labs/causalToolbox@", commit),
      upgrade = "never"
    )
  }

  source_dir <- checkout_source("forestry-labs/causalToolbox", commit, "tcg-causalToolbox-")
  srf_path <- file.path(source_dir, "R", "SRF.R")
  if (!file.exists(srf_path)) stop("Missing SRF.R in locked causalToolbox source")

  srf <- paste(readLines(srf_path, warn = FALSE), collapse = "\n")
  old_call <- paste(
    '      splitratio = hyperparameter_list[["mu.forestry"]]$splitratio,',
    '      OOBhonest = hyperparameter_list[["mu.forestry"]]$OOBhonest',
    sep = "\n"
  )
  new_call <- paste(
    '      splitratio = hyperparameter_list[["mu.forestry"]]$splitratio',
    "      # Rforestry 0.10.1 removed the unsupported OOBhonest argument.",
    sep = "\n"
  )
  matches <- gregexpr(old_call, srf, fixed = TRUE)[[1]]
  if (matches[1] < 0 || length(matches) != 1) {
    stop("Expected one S_RF Rforestry call for the causalToolbox compatibility patch")
  }
  srf <- sub(old_call, new_call, srf, fixed = TRUE)
  old_subset <- 'feat[, hyperparameter_list[["mu.forestry"]]$relevant.Variable]'
  new_subset <- 'feat[, hyperparameter_list[["mu.forestry"]]$relevant.Variable, drop = FALSE]'
  matches <- gregexpr(old_subset, srf, fixed = TRUE)[[1]]
  if (matches[1] < 0 || length(matches) != 1) {
    stop("Expected one S_RF feature subset for the causalToolbox dimension-preservation patch")
  }
  writeLines(sub(old_subset, new_subset, srf, fixed = TRUE), srf_path)

  description_path <- file.path(source_dir, "DESCRIPTION")
  description <- paste(readLines(description_path, warn = FALSE), collapse = "\n")
  description <- sub("\nRemotes: forestry-labs/Rforestry", "", description, fixed = TRUE)
  writeLines(description, description_path)
  stamp_source(source_dir, commit, patch)
  remotes::install_local(source_dir, dependencies = FALSE, upgrade = "never", force = TRUE)
  unlink(source_dir, recursive = TRUE)
}

install_rforestry_compat <- function(commit) {
  patch <- "unknown-core-count-v1"
  if (installed_source_matches("Rforestry", commit, patch)) {
    checker <- paste(
      deparse(body(get("training_data_checker", asNamespace("Rforestry")))),
      collapse = " "
    )
    if (grepl("isTRUE", checker, fixed = TRUE)) {
      message("Rforestry core-count compatibility patch is already installed")
      return(invisible(NULL))
    }
  }
  source_dir <- checkout_source("forestry-labs/Rforestry", commit, "tcg-Rforestry-")
  forestry_path <- file.path(source_dir, "R", "forestry.R")
  if (!file.exists(forestry_path)) stop("Missing forestry.R in locked Rforestry source")

  forestry <- paste(readLines(forestry_path, warn = FALSE), collapse = "\n")
  old_check <- "nthread > parallel::detectCores(),"
  new_check <- "isTRUE(nthread > parallel::detectCores()),"
  matches <- gregexpr(old_check, forestry, fixed = TRUE)[[1]]
  if (matches[1] < 0 || length(matches) != 1) {
    stop("Expected one Rforestry core-count check for the compatibility patch")
  }
  writeLines(sub(old_check, new_check, forestry, fixed = TRUE), forestry_path)
  stamp_source(source_dir, commit, patch)
  remotes::install_local(source_dir, dependencies = FALSE, upgrade = "never", force = TRUE)
  unlink(source_dir, recursive = TRUE)
}

install_causal_toolbox("75acc74ae689fddebe63a43597ff21f3ec233bd4")
install_rforestry_compat("92b396d77d7ce2bbaba1c6d97c8861ea43ae1566")
