# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using EmpiricalDistributions

# Doctest setup
DocMeta.setdocmeta!(
    EmpiricalDistributions,
    :DocTestSetup,
    :(using EmpiricalDistributions);
    recursive=true,
)

makedocs(
    sitename = "EmpiricalDistributions",
    modules = [EmpiricalDistributions],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://oschulz.github.io/EmpiricalDistributions.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    warnonly = ("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/oschulz/EmpiricalDistributions.jl.git",
    forcepush = true,
    push_preview = true,
)
