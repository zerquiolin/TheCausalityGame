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
      install.packages(package, repos = "https://cloud.r-project.org")
    } else {
      remotes::install_version(package, version = version, repos = "https://cloud.r-project.org")
    }
  }
}

install_cran("remotes", "2.5.0")
install_cran("jsonlite", "1.9.1")
remotes::install_github(
  "susanathey/causalTree@51a8762bff0ab17b971bce852cb37bece2d8240f",
  upgrade = "never"
)
remotes::install_github(
  "forestry-labs/causalToolbox@75acc74ae689fddebe63a43597ff21f3ec233bd4",
  upgrade = "never"
)
