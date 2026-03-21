using BranchingFlows
using Documenter

DocMeta.setdocmeta!(BranchingFlows, :DocTestSetup, :(using BranchingFlows); recursive=true)

makedocs(;
    modules=[BranchingFlows],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="BranchingFlows.jl",
    checkdocs = :none,
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/BranchingFlows.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Flowception" => "flowception.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/BranchingFlows.jl",
    devbranch="main",
)
