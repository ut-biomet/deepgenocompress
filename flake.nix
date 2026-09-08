{
  description = "Flake for a python environment (deps managed without nix)";
  inputs.nixpkgs.url = "nixpkgs/nixos-26.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos-cuda.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

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

        cyvcf2 = pyPkgs.callPackage ./nix_pkgs/cyvcf2.nix { };
        deepgenocompress = pyPkgs.callPackage ./nix_pkgs/deepgenocompress.nix { inherit cyvcf2; };
        deepgenocompress_no_tests =
          (pyPkgs.callPackage ./nix_pkgs/deepgenocompress.nix { inherit cyvcf2; }).overrideAttrs
            (oldAttrs: {
              doCheck = false;
              doInstallCheck = false;
            });

      in
      {
        devShells.default = pkgs.mkShell rec {
          XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}";

          # trouble with ruff, NixOS cannot run dynamically linked executables
          # do not install them an use the nix packages instead
          UV_NO_GROUP = "lint_lsp_formatter";
          QUARTO_PYTHON = "python"; # make quarto uses .venv's python

          buildInputs = with pkgs; [
            bashInteractive
            pyPkgs.python
            pyPkgs.venvShellHook

            uv

            cudaPackages.cudatoolkit
            cudaPackages.cudnn

            # code formatter / linter / lsp
            nixfmt
            isort
            black
            ruff
            pyPkgs.docformatter
            basedpyright

            # doc
            quarto

            # some shared libraries needed (uv/poetry etc... do not install them)
            stdenv.cc.cc.lib
            zlib # for numpy
          ];
          venvDir = "./.venv";

          postShellHook = ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath (
                [
                  "/run/opengl-driver" # Needed to find cuda related `.so`
                ]
                ++ buildInputs
              )
            }:$LD_LIBRARY_PATH"
            uv sync --no-dev --group dev-no-exec
          '';
        };

        devShells.nix_build_env = pkgs.mkShell {
          # environment with the nix built package
          buildInputs = with pkgs; [
            bashInteractive
            pyPkgs.python
            (python.withPackages (
              ps: with ps; [
                deepgenocompress_no_tests # to get faster build
                tensorflowWithCuda # explicitly set this to get cuda suport
              ]
            ))
          ];
        };
        packages.deepgenocompress = deepgenocompress;
      }
    );
}
