# install Julia and make sure it's in the PATH of the current shell
curl -fsSL https://install.julialang.org | sh -s -- --yes 

# Julia startup file
mkdir -p ~/.julia/config
cp .devcontainer/julia_startup.jl ~/.julia/config/startup.jl

# Github CLI autocomplete (https://www.ajfriesen.com/github-cli-auto-completion-with-oh-my-zsh/)
mkdir -p ~/.oh-my-zsh/completions
/usr/bin/gh completion -s zsh > ~/.oh-my-zsh/completions/_gh
echo "autoload -U compinit" >> ~/.zshrc
echo "compinit -i" >> ~/.zshrc

# Install Julia packages, registries, ...
/home/vscode/.juliaup/bin/julia .devcontainer/onCreate.jl

# Fix permissions for Julia Base, stdlibs (see https://github.com/JuliaLang/juliaup/issues/865, https://github.com/JuliaLang/julia/pull/51150/files)
find /home/vscode/.julia/juliaup/*/share/julia/{base,stdlib} -type f -name '*.jl' -exec chmod 444 {} \;
