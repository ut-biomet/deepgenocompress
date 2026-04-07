{
  description = "Flake for a python environment (deps managed without nix)";
  inputs.nixpkgs.url = "nixpkgs/nixos-25.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python3;
        pyPkgs = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell rec {
          XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}";

          buildInputs = with pkgs; [
            bashInteractive
            pyPkgs.python
            pyPkgs.venvShellHook

            uv

            cudaPackages.cudatoolkit
            cudaPackages.cudnn

            # code formatter / linter / lsp
            nixfmt-rfc-style
            isort
            black
            ruff
            basedpyright

            # some shared libraries needed (uv/poetry etc... do not install them)
            stdenv.cc.cc.lib
            zlib # for numpy
            linuxPackages.nvidia_x11 # for cuda/tensorflow
          ];
          venvDir = "./.venv";
          postVenvCreation = ''
            uv sync
          '';

          postShellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
            uv sync
          '';
        };
      }
    );
}
