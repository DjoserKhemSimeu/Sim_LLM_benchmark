{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      jupyter
      pandas
      numpy
      matplotlib
      seaborn
      scipy
      # Ajoutez ici d'autres dépendances si nécessaire
    ]))
  ];

  shellHook = ''
    echo "Environnement Nix prêt pour Jupyter."
  '';
}
