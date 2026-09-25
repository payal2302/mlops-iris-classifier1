# DVC Workflow

## Local DVC Remote

A local DVC remote was configured using the `dvc-remote-storage` folder.

## DVC Workflow

For each dataset change, the following workflow was used:

1. Generate or modify the dataset.
2. Run `dvc add` to track the dataset with DVC.
3. Run `git add` to stage the DVC metadata file.
4. Run `git commit` to save the dataset version in Git.
5. Run `dvc push` to store the actual dataset in the DVC remote.

## Version Comparison

`dvc diff` was used to compare different dataset versions and identify changes.

## Version Restoration

`git checkout` was used to select the required DVC metadata version, and `dvc checkout` was used to restore the corresponding dataset.

The original dataset contained 150 rows, while the augmented dataset contains 170 rows.