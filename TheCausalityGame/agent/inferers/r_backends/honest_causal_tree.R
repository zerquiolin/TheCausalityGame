all_args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", all_args, value = TRUE)[[1]]
Sys.setenv(TCG_R_BACKEND_METHOD = "honest_causal_tree")
source(file.path(dirname(normalizePath(sub("^--file=", "", script_arg))), "cate_backend.R"))
